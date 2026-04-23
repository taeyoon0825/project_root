from __future__ import annotations

"""메타데이터에 기록된 오디오 자산을 Whisper로 일괄 전사한다.

STT 단계는 단순히 텍스트만 만드는 역할이 아니다. 출력 경로를 결정하고,
세그먼트 부가 파일을 만들고, CSV를 만들고, 메타데이터 상태를 갱신해
파이프라인의 나머지 단계가 전사 결과를 완성된 아티팩트 집합으로 다룰 수 있게 한다.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from src.adaptive.parameter_resolver import build_adaptive_context
import pandas as pd

from src.config import AUDIO_STT_DIR, DEFAULT_METADATA_CSV, STT_CSV_DIR, WHISPER_CACHE_DIR, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame, save_metadata_frame
from src.utils.device import resolve_stt_device, stt_gpu_required
from src.utils.io_utils import save_json

try:
    import whisper
except ImportError:  # pragma: no cover
    whisper = None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _safe_stem(value: str) -> str:
    """ID를 파일 시스템에서 안전한 전사 파일명으로 바꾸기 전에 정리한다."""
    cleaned = re.sub(r"[^0-9A-Za-z_\-]+", "_", value.strip())
    return cleaned.strip("._") or "transcript"


def _safe_print(message: str) -> None:
    """Windows 콘솔에서 인코딩 혼합으로 출력이 깨지지 않도록 방어적으로 출력한다."""
    encoding = sys.stdout.encoding or "utf-8"
    print(str(message).encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _row_source_type(row) -> str:
    """메타데이터 행 dtype이 섞여 있어도 source_type 접근 결과를 일관되게 만든다."""
    return str(row.get("source_type", "")).strip()


def _resolve_audio_path(row) -> Path:
    """레거시 행이 여러 컬럼명을 쓸 수 있으므로 오디오 경로 조회 규칙을 중앙화한다."""
    value = str(row.get("audio_path", "")).strip() or str(row.get("audio_file_path", "")).strip()
    return Path(value)


def _resolve_stt_output_path(row, row_index: int) -> Path:
    """전사 텍스트 출력 경로 규칙을 한곳에서 결정한다.

    재시도, mp4 파생 행, fallback ID가 모두 같은 경로 규칙을 따르려면
    배치 루프 여기저기에 경로 로직을 흩뿌리지 않고 이 함수가 규칙을 소유해야 한다.
    """
    explicit_stt_path = str(row.get("stt_txt_path", "")).strip()
    if explicit_stt_path:
        return Path(explicit_stt_path)

    source_type = _row_source_type(row)
    if source_type in {"youtube_mp4", "youtube_wav"}:
        explicit_processed_path = str(row.get("processed_txt_path", "")).strip()
        if explicit_processed_path:
            return Path(explicit_processed_path)

    doc_id = str(row.get("id", "")).strip() or f"row_{row_index + 1:04d}"
    return AUDIO_STT_DIR / f"{_safe_stem(doc_id)}.txt"


def _normalize_language(language: str | None) -> str | None:
    """빈 값이나 'auto' 언어 설정을 Whisper API가 이해하는 None으로 바꾼다."""
    if language is None:
        return None
    normalized = str(language).strip().lower()
    if not normalized or normalized == "auto":
        return None
    return normalized


def _segments_output_path(stt_txt_path: Path) -> Path:
    """문장 세그먼트 타이밍 payload를 본문 전사 파일 옆에 둔다."""
    return stt_txt_path.with_suffix(".segments.json")


def _stt_csv_output_path(stt_txt_path: Path) -> Path:
    """세그먼트 표 형식 내보내기 파일을 전용 STT CSV 디렉터리에 쓴다."""
    return STT_CSV_DIR / f"{stt_txt_path.stem}.csv"


def _write_stt_csv(stt_txt_path: Path, transcript: str, language: str | None, segments: list[dict] | None) -> Path:
    """STT 결과를 분석과 디버깅에 쓰기 쉬운 CSV 형태로 물질화한다.

    원문 텍스트와 JSON 부가 파일만으로는 다루기 불편할 수 있으므로,
    하위 도구가 Whisper 원본 payload 구조를 몰라도 세그먼트를 볼 수 있게 한다.
    """
    STT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _stt_csv_output_path(stt_txt_path)
    rows = []
    if segments:
        for index, segment in enumerate(segments, start=1):
            rows.append(
                {
                    "segment_index": index,
                    "start": float(segment.get("start", 0.0) or 0.0),
                    "end": float(segment.get("end", 0.0) or 0.0),
                    "text": str(segment.get("text", "") or "").strip(),
                    "language": str(language or ""),
                }
            )
    if not rows:
        rows = [{"segment_index": 1, "start": 0.0, "end": 0.0, "text": str(transcript or ""), "language": str(language or "")}]
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def transcribe_audio_batch(
    metadata_path: Path = DEFAULT_METADATA_CSV,
    model_name: str | None = None,
    language: str | None = None,
    overwrite: bool = False,
    source_type: str | None = None,
    skip_errors: bool = True,
    target_ids: set[str] | None = None,
) -> Path:
    """선택된 메타데이터 행을 전사하고 관련 STT 아티팩트를 모두 저장한다.

    파이프라인은 STT를 순수 추론 헬퍼가 아니라 상태를 가진 일괄 처리 단계로 본다.
    그래서 선택, Whisper 호출, 파일 저장, 메타데이터 갱신을 이 함수가 한꺼번에 담당한다.
    """
    if whisper is None:
        raise ImportError("openai-whisper is not installed. Run `pip install openai-whisper`.")

    ensure_project_dirs()
    metadata = load_metadata_frame(metadata_path)
    candidate_indices: list[int] = []
    # 실제 작업 대상 목록을 먼저 만들어야
    # 전사할 파일이 없을 때 디바이스/모델 초기화를 하지 않는다.
    for row_index, row in metadata.iterrows():
        row_id = str(row.get("id", "")).strip()
        row_source_type = _row_source_type(row)
        if source_type and row_source_type != source_type and not (row_source_type == "youtube_wav" and source_type == "youtube_mp4"):
            continue
        if target_ids and row_id not in target_ids:
            continue
        candidate_indices.append(row_index)

    if not candidate_indices:
        _safe_print("[STT] No target files selected. STT skipped.")
        return metadata_path

    subset = metadata.iloc[candidate_indices].reset_index(drop=True)
    # 적응형 기본값은 실제 처리 대상 부분집합으로부터 뽑아야
    # 모델/언어 선택이 과거 전체 메타데이터가 아니라 현재 배치에 맞춰진다.
    adaptive_context = build_adaptive_context(subset, text_source="stt_transcript")
    resolved_model_name = str(model_name or adaptive_context.language.whisper_model or "base").strip() or "base"
    whisper_language = _normalize_language(language)
    if whisper_language is None:
        whisper_language = adaptive_context.language.whisper_language

    device = resolve_stt_device()
    _safe_print(f"[STT] dedicated device={device}")
    model = whisper.load_model(resolved_model_name, download_root=str(WHISPER_CACHE_DIR), device=device)
    if stt_gpu_required() and not str(device).startswith("cuda"):
        raise RuntimeError(f"STT requires CUDA-only execution, but resolved device={device!r}.")
    total = len(candidate_indices)

    for progress_index, row_index in enumerate(candidate_indices, start=1):
        row = metadata.iloc[row_index]
        row_source_type = _row_source_type(row)
        audio_path = _resolve_audio_path(row)
        stt_txt_path = _resolve_stt_output_path(row, row_index)
        stt_txt_path.parent.mkdir(parents=True, exist_ok=True)

        _safe_print(f"[stt {progress_index}/{total}] {row.get('file_name', row.get('id', 'unknown'))}")
        if not audio_path.exists():
            message = f"Audio file does not exist: {audio_path}"
            metadata.at[row_index, "processing_status"] = "stt_audio_missing"
            metadata.at[row_index, "error_message"] = message
            if skip_errors:
                _safe_print(f"  - skipped: {message}")
                continue
            raise FileNotFoundError(message)

        try:
            if overwrite or not stt_txt_path.exists():
                # 아티팩트가 없거나 명시적으로 무효화된 경우에만 Whisper를 호출한다.
                # STT는 비용이 큰 단계이므로 기본값은 캐시 재사용이다.
                result = model.transcribe(
                    str(audio_path),
                    language=whisper_language,
                    fp16=device.startswith("cuda"),
                    condition_on_previous_text=False,
                    verbose=False,
                )
                transcript = str(result.get("text", "")).strip()
                stt_txt_path.write_text(transcript, encoding="utf-8")
                save_json(
                    _segments_output_path(stt_txt_path),
                    {
                        "text": transcript,
                        "language": result.get("language"),
                        "segments": result.get("segments", []),
                    },
                )
                stt_csv_path = _write_stt_csv(
                    stt_txt_path,
                    transcript=transcript,
                    language=str(result.get("language") or whisper_language or ""),
                    segments=result.get("segments", []),
                )
                _safe_print(f"  - transcribed to {stt_txt_path}")
            else:
                # 기존 전사문을 재사용하더라도 세그먼트 부가 파일과 CSV는 다시 맞춰 준다.
                # 그래야 옛 실행 결과도 재추론 없이 최신 아티팩트 구조를 얻는다.
                transcript = stt_txt_path.read_text(encoding="utf-8").strip()
                segments_path = _segments_output_path(stt_txt_path)
                if not segments_path.exists():
                    save_json(
                        segments_path,
                        {
                            "text": transcript,
                            "language": whisper_language,
                            "segments": [],
                        },
                    )
                result = {"language": whisper_language, "segments": []}
                try:
                    payload = segments_path.read_text(encoding="utf-8")
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        result["segments"] = parsed.get("segments", []) or []
                except Exception:
                    result["segments"] = []
                stt_csv_path = _write_stt_csv(
                    stt_txt_path,
                    transcript=transcript,
                    language=str(result.get("language") or whisper_language or ""),
                    segments=result.get("segments", []),
                )
                _safe_print(f"  - reused existing transcript {stt_txt_path}")

            metadata.at[row_index, "stt_transcript"] = transcript
            metadata.at[row_index, "stt_txt_path"] = str(stt_txt_path.resolve())
            metadata.at[row_index, "stt_csv_path"] = str(stt_csv_path.resolve())
            metadata.at[row_index, "stt_model_name"] = resolved_model_name
            metadata.at[row_index, "stt_device"] = device
            metadata.at[row_index, "stt_language"] = str(result.get("language") or whisper_language or "")
            metadata.at[row_index, "adaptive_whisper_language"] = whisper_language or ""
            metadata.at[row_index, "adaptive_language_reason"] = adaptive_context.language.reasoning
            metadata.at[row_index, "error_message"] = ""

            if row_source_type in {"youtube_mp4", "youtube_wav"}:
                # 레거시 하위 로직은 media 파생 행에서 processed_txt_path가
                # STT 출력 경로를 가리킨다고 가정하므로 alias를 함께 맞춰 준다.
                metadata.at[row_index, "processed_txt_path"] = str(stt_txt_path.resolve())
                metadata.at[row_index, "processing_status"] = "transcribed"
            else:
                metadata.at[row_index, "processing_status"] = "stt_completed"
        except Exception as exc:
            metadata.at[row_index, "processing_status"] = "stt_error"
            metadata.at[row_index, "error_message"] = str(exc)
            if skip_errors:
                _safe_print(f"  - failed: {exc}")
                continue
            raise
        finally:
            # 긴 배치 실행에서는 GPU 메모리 단편화가 누적될 수 있으므로
            # 파일 사이마다 캐시를 비워 루프 안정성을 높인다.
            if torch is not None and device.startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    save_metadata_frame(metadata, metadata_path)
    return metadata_path


def main() -> None:
    """프로젝트 전체 배치 전사를 위한 CLI 진입점."""
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-type", type=str, default=None)
    parser.add_argument("--skip-errors", dest="skip_errors", action="store_true")
    parser.add_argument("--no-skip-errors", dest="skip_errors", action="store_false")
    parser.set_defaults(skip_errors=True)
    args = parser.parse_args()

    output_path = transcribe_audio_batch(
        metadata_path=args.metadata_path,
        model_name=args.model_name,
        language=args.language,
        overwrite=args.overwrite,
        source_type=args.source_type,
        skip_errors=args.skip_errors,
    )
    _safe_print(f"STT transcription completed. Metadata updated at {output_path}")


if __name__ == "__main__":
    main()
