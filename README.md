# Qué incluye este repositorio

README.md
cascada_TFM
  - run_cascade_escucha1.py: pipeline cascada (3 fases: transcripcion, respuesta, scoring)
  - fix_hallucinations.py: control de calidad: detecta y corrige alucinaciones de ASR
  - verifiers_aif.py: 33 verificadores de seguimiento de instrucciones (items aif)
Voxtral
  - run_voxtral_audio_native.py: evaluacion audio-nativa (Voxtral-Mini-3B, sin cascada)
dataset/
  - ESCUCHA1.json: dataset final, con todas las transcripciones ya generadas
  - ESCUCHA1.orig.json: dataset tal como se recibio, antes de ejecutar nada (para diff/auditoria)
results/
  - cascade-salamandra-escucha1-responses.json: respuestas de la cascada (1000/1000)
  - voxtral-audio-native-responses.json: respuestas de Voxtral (1000/1000)
logs/
 - cascade_run.log: log completo de la ejecucion principal de la cascada
 - Fix_hallucinations.log: log de la correccion de alucinaciones
