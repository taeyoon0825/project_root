from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
AUX_CACHE_DIR = PROJECT_ROOT / ".cache"
WHISPER_CACHE_DIR = AUX_CACHE_DIR / "whisper"
HF_CACHE_DIR = AUX_CACHE_DIR / "huggingface"
AUDIO_DIR = DATA_DIR / "audio"
AUDIO_MP4_DIR = AUDIO_DIR / "mp4"
AUDIO_WAV_DIR = AUDIO_DIR / "wav"
AUDIO_TMP_DIR = AUDIO_DIR / "tmp"
AUDIO_STT_DIR = AUDIO_DIR / "stt_txt"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
HTML_UPLOADS_DIR = DATA_DIR / "html_uploads"
HTML_UPLOAD_MEDIA_DIR = HTML_UPLOADS_DIR / "media"
HTML_UPLOAD_WAV_DIR = HTML_UPLOADS_DIR / "wav"
HTML_UPLOAD_TRANSCRIPTS_DIR = HTML_UPLOADS_DIR / "transcripts"

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

DEFAULT_METADATA_CSV = METADATA_DIR / "dataset_metadata.csv"
REALDATA_METADATA_CSV = METADATA_DIR / "youtube_mp4_metadata.csv"
COMBINED_METADATA_CSV = METADATA_DIR / "combined_dataset_metadata.csv"
DEFAULT_QUERYSET_CSV = METADATA_DIR / "evaluation_queries.csv"
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
    ]:
        directory.mkdir(parents=True, exist_ok=True)
