# Evaluación audio-nativa (Voxtral) en Google Colab

## 0. Requisito de GPU

Se necesita **GPU L4** (24GB) como mínimo — requiere **Colab Pro** o
créditos 

## 1. Qué subir a Google Drive

En una carpeta (p. ej. `colab_voxtral/`):
- `run_voxtral_audio_native.py`
- `ESCUCHA1.json`
- carpeta `audio/` (2.7GB si es todo el dataset; si solo faltan algunos
  ítems, basta con esos)
- si ya hay resultados previos, `results/voxtral-audio-native-responses.json`
  (el script reanuda automáticamente y no repite lo ya hecho)

## 2. Preparar el entorno de ejecución de Colab

**Entorno de ejecución → Cambiar tipo de entorno de ejecución**: **GPU L4**

## 3. Ejecutar Voxtral_ESCUCHA.ipynb
