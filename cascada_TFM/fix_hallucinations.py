
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_cascade_escucha1 import (
    DEFAULT_DATASET, DEFAULT_OUTPUT, WHISPER_REPO, LLM_PATH,
    load_dataset, save_dataset, load_json_dict, save_json,
    pending_transcriptions, extract_segment,
    build_prompt, fit_transcripts, STOP_STRINGS,
)

AFECTADOS = json.load(open("/tmp/afectados_hallucinacion.json"))

def refazer_transcripciones():
    import static_ffmpeg
    static_ffmpeg.add_paths()
    import shutil
    import tempfile
    import mlx_whisper
    from collections import defaultdict

    ffmpeg_bin = shutil.which("ffmpeg")
    items = load_dataset(DEFAULT_DATASET)
    by_id = {it["id"]: it for it in items}

    grupos = defaultdict(list)
    for item_id, campo in AFECTADOS:
        texto_actual = by_id[item_id]["aux_metadata"][campo]
        grupos[(campo, texto_actual)].append(item_id)

    print(f"{len(AFECTADOS)} items afectados agrupados en {len(grupos)} audios unicos a retranscribir")

    for n, ((campo, _texto_viejo), ids) in enumerate(grupos.items(), 1):
        rep_id = ids[0]
        rep_item = by_id[rep_id]
        aux = rep_item["aux_metadata"]
        is_dual = bool(aux.get("url_2"))
        dur1 = float(aux["actual_duration_1"]) if aux.get("actual_duration_1") else None
        audio_path = Path("audio") / f"{rep_id}.mp3"

        if campo == "transcription_1" and is_dual and dur1:
            start, end = 0.0, dur1
        else:
            start, end = None, None

        kwargs = dict(path_or_hf_repo=WHISPER_REPO, language="es",
                      condition_on_previous_text=False)
        if start is None:
            text = mlx_whisper.transcribe(str(audio_path), **kwargs)["text"].strip()
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                extract_segment(ffmpeg_bin, audio_path, start, end, tmp.name)
                text = mlx_whisper.transcribe(tmp.name, **kwargs)["text"].strip()

        for item_id in ids:
            by_id[item_id]["aux_metadata"][campo] = text
        print(f"[{n}/{len(grupos)}] {len(ids)} items <- {rep_id}/{campo}: "
              f"{len(text)} chars | {text[:100]!r}")
        save_dataset(items, DEFAULT_DATASET)

    print(f"\nDataset actualizado: {DEFAULT_DATASET}")

def regenerar_respuestas():
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_logits_processors

    items = load_dataset(DEFAULT_DATASET)
    by_id = {it["id"]: it for it in items}
    affected_ids = {i for i, c in AFECTADOS}

    results = load_json_dict(DEFAULT_OUTPUT)
    by_result_id = {r["id"]: r for r in results}

    print(f"Cargando LLM: {LLM_PATH}")
    model, tokenizer = load(str(LLM_PATH))
    logits_processors = make_logits_processors(repetition_penalty=1.2, repetition_context_size=20)

    for n, item_id in enumerate(sorted(affected_ids), 1):
        item = by_id[item_id]
        aux = item["aux_metadata"]
        t1, t2 = aux.get("transcription_1"), aux.get("transcription_2")
        eval_type = item.get("eval_type", "mcqa")
        max_tokens = 64 if eval_type == "mcqa" else 256

        t1c, t2c, truncated = fit_transcripts(tokenizer, item, t1, t2, max_tokens)
        user = build_prompt(item, t1c, t2c)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
        output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False,
                          logits_processors=logits_processors)
        for stop in STOP_STRINGS:
            output = output.split(stop, 1)[0]
        output = output.strip()

        old_output = by_result_id.get(item_id, {}).get("output")
        by_result_id[item_id] = {**item, "output": output}
        print(f"[{n}/{len(affected_ids)}] {item_id}: {old_output!r} -> {output!r}")

    results = list(by_result_id.values())
    save_json(results, DEFAULT_OUTPUT)
    print(f"\nResultados actualizados: {DEFAULT_OUTPUT}")

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("transcribe", "all"):
        refazer_transcripciones()
    if stage in ("answer", "all"):
        regenerar_respuestas()
