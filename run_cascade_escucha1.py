import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_DATASET = HERE / "ESCUCHA1.json"
DEFAULT_AUDIO_ROOT = HERE / "audio"
DEFAULT_OUTPUT = HERE / "results" / "cascade-salamandra-escucha1-responses.json"
WHISPER_REPO = "mlx-community/whisper-large-v3-mlx"
LLM_PATH = HERE / "models" / "salamandra-2b-instruct-mlx-4bit"
STOP_STRINGS = ["<|im_end|>", "</s>"]
MAX_CONTEXT = 8192
CONTEXT_MARGIN = 200

def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_dataset(items, path, retries=3, delay=2.0):
    path = Path(path)
    for attempt in range(1, retries + 1):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            return
        except OSError as e:
            print(f"aviso: fallo al guardar {path} (intento {attempt}/{retries}): {e}", file=sys.stderr)
            time.sleep(delay)
    print(f"ERROR: no se pudo guardar {path} tras {retries} intentos; se continua sin persistir este checkpoint",
          file=sys.stderr)

def load_json_dict(path):
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path, retries=3, delay=2.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            return
        except OSError as e:
            print(f"aviso: fallo al guardar {path} (intento {attempt}/{retries}): {e}", file=sys.stderr)
            time.sleep(delay)
    print(f"ERROR: no se pudo guardar {path} tras {retries} intentos; se continua sin persistir este checkpoint",
          file=sys.stderr)

def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def pending_transcriptions(item):
    aux = item["aux_metadata"]
    is_dual = bool(aux.get("url_2"))
    dur1 = _to_float(aux.get("actual_duration_1"))
    todo = []
    if not aux.get("transcription_1"):
        if is_dual and dur1:
            todo.append(("transcription_1", 0.0, dur1))
        else:
            todo.append(("transcription_1", None, None))
    if is_dual and not aux.get("transcription_2"):
        todo.append(("transcription_2", dur1, None))
    return todo

def extract_segment(ffmpeg_bin, src, start, end, dst):
    cmd = [ffmpeg_bin, "-y", "-i", str(src)]
    if start:
        cmd += ["-ss", str(start)]
    if end:
        cmd += ["-to", str(end)]
    cmd += ["-ar", "16000", "-ac", "1", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exit {proc.returncode}: {proc.stderr[-500:]}")

def stage_transcribe(args):
    import static_ffmpeg
    static_ffmpeg.add_paths()
    import shutil
    import mlx_whisper

    ffmpeg_bin = shutil.which("ffmpeg")

    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]

    work = [(it, pending_transcriptions(it)) for it in items]
    n_pending = sum(len(todo) for _, todo in work)
    n_reused = sum(
        1 for it in items for f in ("transcription_1", "transcription_2")
        if it["aux_metadata"].get(f)
    )
    print(f"Transcribir: {len(items)} items | clips ya presentes (reutilizados): {n_reused} | "
          f"clips pendientes: {n_pending}")

    done_count = 0
    for item, todo in work:
        if not todo:
            continue
        audio_path = Path(args.audio_root) / f"{item['id']}.mp3"
        if not audio_path.exists():
            print(f"falta audio: {audio_path}", file=sys.stderr)
            continue

        for field, start, end in todo:
            t0 = time.time()
            try:
                if start is None:
                    text = mlx_whisper.transcribe(
                        str(audio_path), path_or_hf_repo=WHISPER_REPO, language="es")["text"].strip()
                else:
                    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                        extract_segment(ffmpeg_bin, audio_path, start, end, tmp.name)
                        text = mlx_whisper.transcribe(
                            tmp.name, path_or_hf_repo=WHISPER_REPO, language="es")["text"].strip()
            except Exception as e:
                print(f"ASR fallo en {item['id']}/{field}: {e}", file=sys.stderr)
                continue
            item["aux_metadata"][field] = text
            done_count += 1
            dt = time.time() - t0
            print(f"[{done_count}/{n_pending}] {item['id']}/{field} ({dt:.0f}s): {text[:80]!r}")

            if done_count % args.checkpoint_every == 0:
                save_dataset(items, args.dataset)

    save_dataset(items, args.dataset)
    print(f"Dataset actualizado: {args.dataset}")

def build_prompt(item, t1, t2):
    lines = [
        "Escucha la siguiente transcripcion de un audio en espanol y responde la pregunta.",
        "Responde SOLO con la respuesta final, sin explicaciones.",
        "",
    ]
    if t2:
        lines += ["Primer audio (transcripcion):", t1 or "(sin transcripcion)", "",
                   "Segundo audio (transcripcion):", t2 or "(sin transcripcion)"]
    else:
        lines += ["Transcripcion del audio:", t1 or "(sin transcripcion disponible)"]
    lines += ["", f"Pregunta: {item['question']}"]
    if item.get("choices"):
        lines.append("Opciones:")
        for i, choice in enumerate(item["choices"], 1):
            lines.append(f"{i}. {choice}")
        lines.append("Responde unicamente con el texto de la opcion elegida.")
    return "\n".join(lines)

def fit_transcripts(tokenizer, item, t1, t2, max_tokens):
    base_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_prompt(item, "", "" if t2 else None)}],
        add_generation_prompt=True, tokenize=False)
    base_tokens = len(tokenizer.encode(base_prompt))
    budget = MAX_CONTEXT - max_tokens - CONTEXT_MARGIN - base_tokens
    if budget <= 0:
        return t1, t2, True

    def cut(text, n):
        ids = tokenizer.encode(text)
        if len(ids) <= n:
            return text, False
        head_n = int(n * 0.6)
        tail_n = n - head_n
        return tokenizer.decode(ids[:head_n]) + "\n[...]\n" + tokenizer.decode(ids[-tail_n:]), True

    if t2:
        b1 = budget // 2
        b2 = budget - b1
        t1c, tr1 = cut(t1, b1)
        t2c, tr2 = cut(t2, b2)
        return t1c, t2c, tr1 or tr2
    else:
        t1c, tr1 = cut(t1, budget)
        return t1c, t2, tr1

def stage_answer(args):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_logits_processors

    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]
    done = load_json_dict(args.output)
    done = {r["id"]: r for r in done} if isinstance(done, list) else {}

    print(f"Cargando LLM: {args.llm_path}")
    model, tokenizer = load(str(args.llm_path))
    logits_processors = make_logits_processors(repetition_penalty=1.2, repetition_context_size=20)

    results = list(done.values())
    pending = [it for it in items if it["id"] not in done]
    print(f"Responder: {len(items)} items | ya hechos: {len(done)} | pendientes: {len(pending)}")
    n_truncated = 0

    for n, item in enumerate(pending, 1):
        aux = item["aux_metadata"]
        t1, t2 = aux.get("transcription_1"), aux.get("transcription_2")
        if not t1 and not t2:
            print(f"[{n}/{len(pending)}] sin transcripcion para {item['id']}; se omite", file=sys.stderr)
            results.append({**item, "output": None})
            continue

        max_tokens = args.mcqa_max_tokens if item.get("eval_type", "mcqa") == "mcqa" else args.open_max_tokens
        t1c, t2c, truncated = fit_transcripts(tokenizer, item, t1, t2, max_tokens)
        if truncated:
            n_truncated += 1
            print(f"[{n}/{len(pending)}] {item['id']}: transcripcion truncada (excede contexto)", file=sys.stderr)

        user = build_prompt(item, t1c, t2c)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
        output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False,
                          logits_processors=logits_processors)
        for stop in STOP_STRINGS:
            output = output.split(stop, 1)[0]
        output = output.strip()

        results.append({**item, "output": output})
        print(f"[{n}/{len(pending)}] {item['id']} -> {output[:80]!r}")

        if n % args.checkpoint_every == 0:
            save_json(results, args.output)

    save_json(results, args.output)
    print(f"Guardado: {args.output} (transcripciones truncadas: {n_truncated})")

def _norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def is_correct(item, output):
    if not output or not item.get("answer"):
        return None
    ans_norm = _norm(item["answer"])
    out_norm = _norm(output)
    if ans_norm and ans_norm in out_norm:
        return True
    m = re.match(r"\s*(\d+)[.)\-:]", output)
    if m and item.get("choices"):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(item["choices"]) and _norm(item["choices"][idx]) == ans_norm:
            return True
    return False

def stage_score(args):
    results = load_json_dict(args.output)
    if not isinstance(results, list):
        print("No hay resultados que puntuar todavia.", file=sys.stderr)
        return

    mcqa = [r for r in results if r.get("eval_type", "mcqa") == "mcqa" and r.get("answer")]
    scored = [(r, is_correct(r, r.get("output"))) for r in mcqa]
    n = len(scored)
    correct = sum(1 for _, c in scored if c)
    skipped = sum(1 for _, c in scored if c is None)
    print(f"mcqa evaluables: {n} | correctas: {correct} | sin output: {skipped} | "
          f"accuracy: {correct / n:.1%}" if n else "sin items evaluables")

    def breakdown(key_fn, label):
        buckets = {}
        for r, c in scored:
            if c is None:
                continue
            k = key_fn(r)
            buckets.setdefault(k, [0, 0])
            buckets[k][1] += 1
            if c:
                buckets[k][0] += 1
        print(f"\nPor {label}:")
        for k, (ok, tot) in sorted(buckets.items()):
            print(f"  {k}: {ok}/{tot} ({ok/tot:.1%})")

    breakdown(lambda r: r.get("length_type", "?"), "length_type")
    breakdown(lambda r: r.get("categories", {}).get("reasoning", "?"), "categories.reasoning")

    from verifiers_aif import verify_aif
    aif = [r for r in results if r.get("eval_type") == "aif"]
    aif_scored = [(r, verify_aif(r, r.get("output"))) for r in aif]
    n_aif = sum(1 for _, c in aif_scored if c is not None)
    correct_aif = sum(1 for _, c in aif_scored if c)
    print(f"\naif evaluables: {n_aif}/{len(aif)} | correctas: {correct_aif} | "
          f"accuracy: {correct_aif / n_aif:.1%}" if n_aif else "\naif: sin items evaluables")

    def breakdown_aif(key_fn, label):
        buckets = {}
        for r, c in aif_scored:
            if c is None:
                continue
            k = key_fn(r)
            buckets.setdefault(k, [0, 0])
            buckets[k][1] += 1
            if c:
                buckets[k][0] += 1
        print(f"\nAIF por {label}:")
        for k, (ok, tot) in sorted(buckets.items()):
            print(f"  {k}: {ok}/{tot} ({ok/tot:.1%})")

    breakdown_aif(lambda r: r["aux_metadata"].get("constraint_category", "?"), "constraint_category")
    breakdown_aif(lambda r: r["aux_metadata"].get("verifier", "?"), "verifier")

    total_evaluables = n + n_aif
    total_correctas = correct + correct_aif
    if total_evaluables:
        print(f"\nGLOBAL (mcqa+aif): {total_correctas}/{total_evaluables} ({total_correctas/total_evaluables:.1%})")

def main():
    parser = argparse.ArgumentParser(description="Cascada local ESCUCHA1 (mlx-whisper + Salamandra-2B MLX)")
    parser.add_argument("--stage", choices=["transcribe", "answer", "score", "all"], required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm-path", type=Path, default=LLM_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mcqa-max-tokens", type=int, default=64)
    parser.add_argument("--open-max-tokens", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    if args.stage in ("transcribe", "all"):
        stage_transcribe(args)
    if args.stage in ("answer", "all"):
        stage_answer(args)
    if args.stage in ("score", "all"):
        stage_score(args)

if __name__ == "__main__":
    main()
