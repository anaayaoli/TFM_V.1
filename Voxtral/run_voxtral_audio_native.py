
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).parent
DEFAULT_DATASET = HERE / "ESCUCHA1.json"
DEFAULT_AUDIO_ROOT = HERE / "audio"
DEFAULT_OUTPUT = HERE / "results" / "voxtral-audio-native-responses.json"
REPO_ID = "mistralai/Voxtral-Mini-3B-2507"
LIMITE_AUDIO_S = 35 * 60

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
    print(f"ERROR: no se pudo guardar {path} tras {retries} intentos", file=sys.stderr)

def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

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

def build_question_text(item):
    lines = [
        "Escucha el audio y responde la pregunta en espanol.",
        "Responde SOLO con la respuesta final, sin explicaciones.",
        "",
        f"Pregunta: {item['question']}",
    ]
    if item.get("choices"):
        lines.append("Opciones:")
        for i, choice in enumerate(item["choices"], 1):
            lines.append(f"{i}. {choice}")
        lines.append("Responde unicamente con el texto de la opcion elegida.")
    return "\n".join(lines)

def prepare_audio_paths(item, audio_root, ffmpeg_bin, tmp_files):
    aux = item["aux_metadata"]
    is_dual = bool(aux.get("url_2"))
    dur1 = _to_float(aux.get("actual_duration_1"))
    src = Path(audio_root) / f"{item['id']}.mp3"

    if not is_dual or not dur1:
        full_dur = _to_float(aux.get("actual_duration_1")) or LIMITE_AUDIO_S + 1
        if full_dur > LIMITE_AUDIO_S:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_files.append(tmp.name)
            extract_segment(ffmpeg_bin, src, 0, LIMITE_AUDIO_S, tmp.name)
            print(f"  (audio recortado a {LIMITE_AUDIO_S}s, duracion original {full_dur:.0f}s)")
            return [tmp.name]
        return [str(src)]

    tmp1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_files += [tmp1.name, tmp2.name]
    extract_segment(ffmpeg_bin, src, 0, min(dur1, LIMITE_AUDIO_S), tmp1.name)
    extract_segment(ffmpeg_bin, src, dur1, None, tmp2.name)
    return [tmp1.name, tmp2.name]

def stage_answer(args):
    import torch
    from transformers import VoxtralForConditionalGeneration, AutoProcessor

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        sys.exit("ffmpeg no encontrado en PATH. Instala ffmpeg (apt-get install ffmpeg "
                 "en Linux, o `pip install static-ffmpeg` y llama a static_ffmpeg.add_paths()).")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float16
    print(f"Device: {device} | dtype: {dtype}")

    items = json.load(open(args.dataset, encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    existing = load_json_dict(args.output)
    existing = existing if isinstance(existing, list) else []
    by_id = {r["id"]: r for r in existing}
    done_ids = {rid for rid, r in by_id.items() if r.get("output")}
    pending = [it for it in items if it["id"] not in done_ids]
    print(f"Items: {len(items)} | ya hechos: {len(done_ids)} | pendientes (incl. reintentos): {len(pending)}")

    print(f"Cargando {REPO_ID} ...")
    processor = AutoProcessor.from_pretrained(REPO_ID)
    if device == "cuda":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Cargando en 8-bit (bitsandbytes) para dejar margen a audios largos...")
        model = VoxtralForConditionalGeneration.from_pretrained(
            REPO_ID, quantization_config=quant_config, device_map="auto")
    else:
        model = VoxtralForConditionalGeneration.from_pretrained(REPO_ID, dtype=dtype, device_map=device)

    for n, item in enumerate(pending, 1):
        audio_path = Path(args.audio_root) / f"{item['id']}.mp3"
        if not audio_path.exists():
            print(f"[{n}/{len(pending)}] falta audio: {audio_path}", file=sys.stderr)
            by_id[item["id"]] = {**item, "output": None}
            continue

        tmp_files = []
        try:
            t0 = time.time()
            audio_paths = prepare_audio_paths(item, args.audio_root, ffmpeg_bin, tmp_files)
            content = [{"type": "audio", "path": p} for p in audio_paths]
            content.append({"type": "text", "text": build_question_text(item)})
            conversation = [{"role": "user", "content": content}]

            max_tokens = 64 if item.get("eval_type", "mcqa") == "mcqa" else 256
            inputs = processor.apply_chat_template(conversation)
            inputs = inputs.to(device, dtype=dtype)
            outputs = model.generate(**inputs, max_new_tokens=max_tokens)
            text = processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            dt = time.time() - t0
        except Exception as e:
            print(f"[{n}/{len(pending)}] FALLO en {item['id']}: {e}", file=sys.stderr)
            text = None
            dt = 0
        finally:
            for f in tmp_files:
                Path(f).unlink(missing_ok=True)
            if device == "cuda":
                torch.cuda.empty_cache()

        by_id[item["id"]] = {**item, "output": text}
        print(f"[{n}/{len(pending)}] ({dt:.1f}s) {item['id']} -> {text!r}")

        if n % args.checkpoint_every == 0:
            save_json(list(by_id.values()), args.output)

    save_json(list(by_id.values()), args.output)
    print(f"Guardado: {args.output}")

def _norm(s):
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

def is_correct(item, output):
    import re
    if not output or not item.get("answer"):
        return None
    ans_norm = _norm(item["answer"])
    out_norm = _norm(output)
    if ans_norm and ans_norm in out_norm:
        return True
    m = re.match(r"\s*(\d+)\s*[.)\-:]?\s*$", output.strip())
    if m and item.get("choices"):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(item["choices"]) and _norm(item["choices"][idx]) == ans_norm:
            return True
    return False

def stage_score(args):
    results = load_json_dict(args.output)
    mcqa = [r for r in results if r.get("eval_type", "mcqa") == "mcqa" and r.get("answer")]
    correctas = sum(1 for r in mcqa if is_correct(r, r.get("output")))
    print(f"mcqa evaluables: {len(mcqa)} | correctas: {correctas} | "
          f"accuracy: {correctas/len(mcqa):.1%}" if mcqa else "sin items evaluables")

def main():
    parser = argparse.ArgumentParser(description="Evaluacion ESCUCHA audio-nativa (Voxtral-Mini-3B)")
    parser.add_argument("--stage", choices=["answer", "score", "all"], default="all")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    if args.stage in ("answer", "all"):
        stage_answer(args)
    if args.stage in ("score", "all"):
        stage_score(args)

if __name__ == "__main__":
    main()
