from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DB_DIR = DATA_DIR / "db"
RAW_DATA_DIR = DATA_DB_DIR / "raw"
STT_CSV_DIR = DATA_DB_DIR / "stt_csv"
JSON_DATA_DIR = DATA_DB_DIR / "json"

# Backward-compatible aliases (gradually migrate callers to *_DATA_DIR names)
RAW_DIR = RAW_DATA_DIR
PROCESSED_DIR = JSON_DATA_DIR / "processed"
METADATA_DIR = JSON_DATA_DIR
AUX_CACHE_DIR = PROJECT_ROOT / ".cache"
WHISPER_CACHE_DIR = AUX_CACHE_DIR / "whisper"
HF_CACHE_DIR = AUX_CACHE_DIR / "huggingface"
AUDIO_DIR = DATA_DB_DIR / "audio"
AUDIO_MP4_DIR = RAW_DATA_DIR / "mp4"
AUDIO_WAV_DIR = RAW_DATA_DIR / "wav"
AUDIO_TMP_DIR = DATA_DB_DIR / "tmp"
AUDIO_STT_DIR = JSON_DATA_DIR / "stt_txt"
TRANSCRIPTS_DIR = JSON_DATA_DIR / "transcripts"
HTML_UPLOADS_DIR = DATA_DB_DIR
HTML_UPLOAD_MEDIA_DIR = RAW_DATA_DIR
HTML_UPLOAD_WAV_DIR = RAW_DATA_DIR / "wav"
HTML_UPLOAD_TRANSCRIPTS_DIR = JSON_DATA_DIR / "transcripts"

YOUTUBE_MP4_INPUT_DIR = AUDIO_MP4_DIR
YOUTUBE_WAV_DIR = AUDIO_WAV_DIR / "youtube_mp4"
YOUTUBE_TRANSCRIPTS_DIR = TRANSCRIPTS_DIR / "youtube_mp4"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
INDICES_DIR = ARTIFACTS_DIR / "indices"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
CLUSTERS_DIR = ARTIFACTS_DIR / "clusters"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"
INGEST_DIR = ARTIFACTS_DIR / "ingest"
ADAPTIVE_DIR = ARTIFACTS_DIR / "adaptive"

DEFAULT_METADATA_CSV = METADATA_DIR / "dataset_metadata.csv"
REALDATA_METADATA_CSV = JSON_DATA_DIR / "youtube_mp4_metadata.csv"
COMBINED_METADATA_CSV = JSON_DATA_DIR / "combined_dataset_metadata.csv"
DEFAULT_QUERYSET_CSV = JSON_DATA_DIR / "evaluation_queries.csv"
PROCESSED_REGISTRY_CSV = INGEST_DIR / "processed_registry.csv"
INCREMENTAL_RUN_SUMMARY_JSON = INGEST_DIR / "incremental_run_summary.json"

EMBEDDING_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
}

OPTIONAL_MODELS = {
    "bge-m3": "BAAI/bge-m3",
}


def ensure_project_dirs() -> None:
    for directory in [
        DATA_DIR,
        DATA_DB_DIR,
        RAW_DATA_DIR,
        STT_CSV_DIR,
        JSON_DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        METADATA_DIR,
        AUX_CACHE_DIR,
        WHISPER_CACHE_DIR,
        HF_CACHE_DIR,
        AUDIO_DIR,
        AUDIO_MP4_DIR,
        AUDIO_WAV_DIR,
        AUDIO_TMP_DIR,
        AUDIO_STT_DIR,
        TRANSCRIPTS_DIR,
        HTML_UPLOADS_DIR,
        HTML_UPLOAD_MEDIA_DIR,
        HTML_UPLOAD_WAV_DIR,
        HTML_UPLOAD_TRANSCRIPTS_DIR,
        YOUTUBE_WAV_DIR,
        YOUTUBE_TRANSCRIPTS_DIR,
        ARTIFACTS_DIR,
        EMBEDDINGS_DIR,
        INDICES_DIR,
        PLOTS_DIR,
        CLUSTERS_DIR,
        EVALUATION_DIR,
        INGEST_DIR,
        ADAPTIVE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
