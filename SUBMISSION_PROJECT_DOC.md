# 프로젝트 재현 가이드

본 문서는 현재 저장소의 실제 코드 기준으로 작성한 재현용 README이다. 목적은 `project_root`를 처음 받은 사람이 동일한 폴더 구조, 실행 순서, 생성 산출물을 최대한 다시 만들 수 있도록 안내하는 데 있다.

중요:

- 이 문서는 `문서만 보고 프로젝트를 이해하는 설명서`가 아니라 `실행 재현 가이드`이다.
- 다만 `Markdown 파일만 단독으로`는 현재 프로젝트를 재현할 수 없다. 재현에는 이 저장소의 실제 코드와, 실데이터 재현 시 원본 MP4/WAV 파일이 필요하다.
- 더미 데이터 파이프라인은 저장소 코드만 있으면 재현 가능하다.
- 실데이터 파이프라인은 동일한 입력 파일을 사용해야 현재와 유사한 결과를 얻을 수 있다.
- 모델 다운로드, TTS, Whisper, 임베딩 모델 로딩은 첫 실행 시 외부 네트워크 상태와 패키지 버전에 영향을 받을 수 있다.

---

## 1. 재현 가능 범위

| 구분 | 재현 가능 여부 | 조건 |
|---|---|---|
| 더미 텍스트 데이터 생성 | 가능 | 저장소 코드 + Python 환경 + 패키지 설치 |
| 더미 TTS/STT 파이프라인 | 가능 | 저장소 코드 + `ffmpeg` + TTS/Whisper 동작 환경 |
| 키워드 검색 / Dense 검색 / 평가 / 시각화 / 클러스터링 | 가능 | 저장소 코드 + 모델 다운로드 가능 환경 |
| HTML 대시보드 실행 | 가능 | 저장소 코드 + 패키지 설치 |
| 현재 저장소와 완전히 동일한 실데이터 결과 | 조건부 가능 | 동일한 MP4/WAV 원본 파일 필요 |
| Markdown 파일만으로 전체 프로젝트 복원 | 불가 | 코드, 모델, 입력 데이터가 별도로 필요 |

---

## 2. 확인한 기준 환경

아래 값은 현재 작업 환경에서 실제 확인한 값이다.

| 항목 | 확인값 |
|---|---|
| OS/쉘 | Windows / PowerShell |
| Python | 3.12.0 |
| ffmpeg | `2026-04-01-git-eedf8f0165-full_build-www.gyan.dev` |
| pandas | 2.3.3 |
| numpy | 2.4.2 |
| scikit-learn | 1.8.0 |
| sentence-transformers | 5.2.2 |
| torch | 2.10.0 |
| openai-whisper | 20250625 |
| plotly | 6.6.0 |
| rank-bm25 | 0.2.2 |
| joblib | 1.5.3 |
| faiss-cpu | 1.13.2 |
| umap-learn | 0.5.11 |
| hdbscan | 0.8.42 |
| edge-tts | 7.2.8 |
| gTTS | 2.5.4 |

패키지 버전은 현재 환경 기준이며, 저장소에는 lock file이 없으므로 다른 환경에서는 세부 결과가 달라질 수 있다.

---

## 3. 필수 준비물

### 3.1 공통

1. 이 저장소 전체 코드
2. Python 3.12 계열 권장
3. `pip`
4. `ffmpeg`가 PATH에 등록된 환경
5. 처음 실행 시 모델 다운로드가 가능한 네트워크

### 3.2 실데이터 재현 시 추가 필요

1. 원본 MP4/WAV 파일
2. 입력 파일을 넣을 경로: `data/audio/mp4/`

---

## 4. 설치 절차

프로젝트 루트에서 아래 순서로 진행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-experiment.txt
ffmpeg -version
```

`ffmpeg -version`이 정상 출력되지 않으면 WAV 변환과 Whisper 디코딩이 실패한다. 실제 코드에서도 `src/audio/audio_utils.py`에서 `ffmpeg`를 필수로 검사한다.

---

## 5. 코드가 자동 생성하는 기본 폴더 구조

`src/config.py`의 `ensure_project_dirs()` 기준으로 아래 폴더가 자동 생성된다.

```text
project_root/
├─ .cache/
│  ├─ whisper/
│  └─ huggingface/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ metadata/
│  ├─ audio/
│  │  ├─ mp4/
│  │  ├─ wav/
│  │  ├─ tmp/
│  │  └─ stt_txt/
│  ├─ transcripts/
│  │  └─ youtube_mp4/
│  └─ html_uploads/
│     ├─ media/
│     ├─ wav/
│     └─ transcripts/
├─ artifacts/
│  ├─ embeddings/
│  ├─ indices/
│  ├─ plots/
│  ├─ clusters/
│  ├─ evaluation/
│  └─ ingest/
└─ web/
   └─ experiment_dashboard.html
```

실데이터 입력 경로는 기본값 기준 `data/audio/mp4/` 이다.

---

## 6. 재현 시나리오 A: 더미 데이터 전체 재생성

이 시나리오는 원본 미디어 파일 없이도 재현 가능하다. 코드가 한국어 장문 transcript를 생성하고, TTS와 Whisper를 거쳐 검색/평가/시각화까지 전부 다시 만든다.

### 6.1 실행 명령

```powershell
python run_audio_experiment_pipeline.py --regenerate-dataset --overwrite-audio --overwrite-stt
```

### 6.2 코드 기준 실제 처리 순서

`run_audio_experiment_pipeline.py` 기준:

1. `data/metadata/dataset_metadata.csv` 생성
2. `data/metadata/evaluation_queries.csv` 생성
3. `data/raw/*.txt` 원문 transcript 생성
4. `data/processed/*.txt` 정규화 텍스트 생성
5. TTS WAV 생성
6. Whisper STT 수행
7. BM25 / TF-IDF 메타데이터 생성
8. Dense 임베딩 및 FAISS 인덱스 생성
9. PCA / t-SNE projection 생성
10. KMeans 클러스터링
11. 평가 결과 저장

### 6.3 기본 파라미터

현재 코드 기본값:

- 총 문서 수: `100`
- TTS provider: `edge`
- Edge voice: `ko-KR-SunHiNeural`
- Whisper model: `base`
- text source: `stt_transcript`, `original_transcript`
- KMeans cluster 수: `6`

### 6.4 재현 후 기대 파일

다음 파일이 생성되면 더미 파이프라인이 정상 수행된 것이다.

#### 데이터

- `data/metadata/dataset_metadata.csv`
- `data/metadata/evaluation_queries.csv`
- `data/raw/doc_001.txt` ~ `doc_100.txt`
- `data/processed/doc_001.txt` ~ `doc_100.txt`
- `data/audio/wav/audio_001.wav` ~ `audio_100.wav`
- `data/audio/stt_txt/DOC-001.txt` 등 STT 결과 텍스트
- `data/audio/stt_txt/DOC-001.segments.json` 등 Whisper segment 정보

#### 검색/인덱스

- `artifacts/indices/keyword_index_metadata__stt_transcript.json`
- `artifacts/indices/keyword_index_metadata__original_transcript.json`
- `artifacts/embeddings/paraphrase-multilingual-MiniLM-L12-v2__stt_transcript_embeddings.npy`
- `artifacts/embeddings/multilingual-e5-base__stt_transcript_embeddings.npy`
- `artifacts/indices/paraphrase-multilingual-MiniLM-L12-v2__stt_transcript.faiss`
- `artifacts/indices/multilingual-e5-base__stt_transcript.faiss`

#### 시각화/클러스터

- `artifacts/plots/*_pca_3d_projection.csv`
- `artifacts/plots/*_tsne_2d_projection.csv`
- `artifacts/plots/*_pca_variance.json`
- `artifacts/clusters/*_kmeans_clusters.csv`
- `artifacts/clusters/*_kmeans_summary.json`

#### 평가

- `artifacts/evaluation/retrieval_eval_detail.csv`
- `artifacts/evaluation/retrieval_eval_summary.csv`
- `artifacts/evaluation/retrieval_eval_source_comparison.csv`
- `artifacts/evaluation/ground_truth_mapping.csv`

### 6.5 최소 검증 명령

```powershell
Get-Item data\metadata\dataset_metadata.csv
Get-Item data\metadata\evaluation_queries.csv
(Get-ChildItem data\raw\*.txt).Count
(Get-ChildItem data\processed\*.txt).Count
(Get-ChildItem data\audio\wav\*.wav).Count
(Get-ChildItem data\audio\stt_txt\*.txt).Count
Get-Item artifacts\evaluation\retrieval_eval_summary.csv
```

`--total-items 100` 기본값으로 실행했다면 `.txt`/`.wav` 개수는 각각 100개여야 한다.

---

## 7. 재현 시나리오 B: 텍스트 전용 더미 실험

오디오와 Whisper 없이 텍스트 기반 검색 실험만 재현하려면 아래 명령을 사용한다.

```powershell
python run_experiment_pipeline.py --total-items 100
```

이 경우:

- `data/raw`, `data/processed`, `data/metadata/*.csv`는 생성된다.
- Dense 검색, 평가, PCA/t-SNE, 클러스터링은 수행된다.
- TTS WAV와 Whisper STT 산출물은 생성되지 않는다.

---

## 8. 재현 시나리오 C: 실제 MP4/WAV 입력 기반 파이프라인

이 시나리오는 실데이터가 있을 때 사용한다. 현재 저장소와 비슷한 실데이터 결과를 재현하려면 입력 파일이 동일해야 한다.

### 8.1 입력 파일 배치

원본 미디어 파일을 아래 경로에 넣는다.

```text
data/audio/mp4/
```

하위 폴더를 사용해도 된다. `run_real_mp4_pipeline.py` 기본값은 재귀 탐색이다.

지원 입력 확장자:

- `.mp4`
- `.wav`

### 8.2 실행 명령

실데이터만 기준으로 재현:

```powershell
python run_real_mp4_pipeline.py --real-only --overwrite-audio --overwrite-stt --show-metrics
```

실데이터와 더미 데이터를 합쳐 검색용 데이터셋을 만들려면:

```powershell
python run_real_mp4_pipeline.py --merge-with-dummy --overwrite-audio --overwrite-stt --show-metrics
```

### 8.3 코드 기준 실제 처리 순서

`run_real_mp4_pipeline.py` 기준:

1. `data/audio/mp4/` 탐색
2. `artifacts/ingest/processed_registry.csv`와 비교하여 증분 대상 계산
3. `data/metadata/youtube_mp4_metadata.csv` 갱신
4. MP4를 WAV로 변환
5. Whisper STT 수행
6. transcript 결과를 metadata에 재반영
7. 필요 시 `combined_dataset_metadata.csv` 생성
8. 키워드 검색 메타데이터 / Dense 임베딩 / FAISS / projection / clustering 생성
9. 평가 결과 저장
10. `incremental_run_summary.json` 기록

### 8.4 재현 후 기대 파일

실데이터 개수에 비례해 아래가 생성된다.

- `data/metadata/youtube_mp4_metadata.csv`
- `data/audio/wav/youtube_mp4/...`
- `data/transcripts/youtube_mp4/*.txt`
- `data/transcripts/youtube_mp4/*.segments.json`
- `artifacts/ingest/processed_registry.csv`
- `artifacts/ingest/incremental_run_summary.json`

평가/인덱스/시각화 산출물은 metadata 파일 stem을 namespace로 사용한다.

예:

- `youtube_mp4_metadata__retrieval_eval_summary.csv`
- `youtube_mp4_metadata__query_catalog.json`
- `youtube_mp4_metadata__multilingual-e5-base__stt_transcript.faiss`
- `youtube_mp4_metadata__multilingual-e5-base__stt_transcript_pca_3d_projection.csv`

### 8.5 결과가 현재 저장소와 다를 수 있는 이유

- 입력 MP4/WAV 파일이 다름
- 파일명/경로가 다르면 stable ID가 달라짐
- Whisper / SentenceTransformer / TTS 패키지 버전 차이
- 첫 실행 시 다운로드되는 모델 버전 차이
- `ffmpeg` 버전 차이

---

## 9. 재현 시나리오 D: HTML 대시보드 실행

HTML 대시보드는 검색, 평가, 문서 비교, 벡터 분포, 클러스터 결과를 한 화면에서 확인하는 실행 경로이다.

### 9.1 실행 명령

```powershell
python html_experiment_app.py
```

브라우저 접속:

```text
http://127.0.0.1:8765
```

### 9.2 현재 코드 기준 동작

- `web/experiment_dashboard.html` 제공
- `/api/options`, `/api/search`, `/api/document`, `/api/plot`, `/api/upload` 엔드포인트 사용
- 검색 문장을 직접 입력하면 현재 메타데이터 전체를 대상으로 검색
- 평가 표에 `precision@k`, `recall@k`, `f1@k`, `accuracy@1_reference` 표시

### 9.3 업로드 재현

HTML 화면에서 MP4/WAV를 업로드하면 아래 경로에 저장된다.

- `data/html_uploads/media/`
- `data/html_uploads/wav/`
- `data/html_uploads/transcripts/`

이 경로의 transcript도 이후 검색 대상에 포함된다.

---

## 10. 현재 코드가 사용하는 주요 입력/출력 경로

`src/config.py` 기준 핵심 경로:

| 항목 | 경로 |
|---|---|
| 더미 metadata | `data/metadata/dataset_metadata.csv` |
| 실데이터 metadata | `data/metadata/youtube_mp4_metadata.csv` |
| 통합 metadata | `data/metadata/combined_dataset_metadata.csv` |
| 평가 질의 | `data/metadata/evaluation_queries.csv` |
| 실데이터 입력 폴더 | `data/audio/mp4/` |
| 실데이터 WAV 출력 | `data/audio/wav/youtube_mp4/` |
| 실데이터 transcript 출력 | `data/transcripts/youtube_mp4/` |
| HTML 업로드 저장 루트 | `data/html_uploads/` |
| Whisper 캐시 | `.cache/whisper/` |
| Hugging Face 캐시 | `.cache/huggingface/` |
| Dense embedding | `artifacts/embeddings/` |
| FAISS index | `artifacts/indices/` |
| PCA/t-SNE 산출물 | `artifacts/plots/` |
| 클러스터 산출물 | `artifacts/clusters/` |
| 평가 산출물 | `artifacts/evaluation/` |
| 증분 처리 기록 | `artifacts/ingest/` |

---

## 11. 동일 결과를 최대한 맞추려면 고정해야 할 요소

정확히 같은 결과를 재현하려면 아래를 함께 고정해야 한다.

1. Python 버전
2. `requirements-experiment.txt` 설치 결과
3. `ffmpeg` 버전
4. Whisper 모델명: 현재 기본 `base`
5. 임베딩 모델 alias:
   - `paraphrase-multilingual-MiniLM-L12-v2`
   - `multilingual-e5-base`
6. 입력 데이터 파일 내용과 파일명
7. 실행 명령 옵션
8. query CSV (`evaluation_queries.csv`)

추가로, 현재 코드 안에서 확인 가능한 고정값은 다음과 같다.

- 더미 데이터 생성 seed: `42`
- KMeans random state: `42`
- KMeans `n_init`: `20`
- E5 계열 모델은 query/passsage prefix를 붙여 임베딩 수행

---

## 12. 재현 체크리스트

아래 항목이 모두 충족되면 재현이 정상적으로 된 것으로 본다.

### 더미 파이프라인

- [ ] `dataset_metadata.csv` 생성
- [ ] `evaluation_queries.csv` 생성
- [ ] `data/raw/*.txt` 100개 생성
- [ ] `data/processed/*.txt` 100개 생성
- [ ] `data/audio/wav/*.wav` 100개 생성
- [ ] `data/audio/stt_txt/*.txt` 생성
- [ ] `artifacts/evaluation/retrieval_eval_summary.csv` 생성
- [ ] `artifacts/indices/*.faiss` 생성
- [ ] `artifacts/plots/*_projection.csv` 생성
- [ ] `artifacts/clusters/*_kmeans_summary.json` 생성

### 실데이터 파이프라인

- [ ] `youtube_mp4_metadata.csv` 생성
- [ ] 입력 파일 수와 metadata row 수가 크게 어긋나지 않음
- [ ] `processed_registry.csv` 생성
- [ ] `incremental_run_summary.json` 생성
- [ ] `youtube_mp4_metadata__retrieval_eval_summary.csv` 생성

### HTML 대시보드

- [ ] `python html_experiment_app.py` 실행 가능
- [ ] `http://127.0.0.1:8765` 접속 가능
- [ ] 검색 문장 직접 입력 가능
- [ ] 평가 요약 표 출력 가능

---

## 13. 한계와 주의사항

1. 이 문서는 저장소 코드를 전제로 한 재현 가이드이다. Markdown 파일 단독으로는 프로젝트를 복원할 수 없다.
2. 실데이터 결과는 입력 MP4/WAV가 없으면 동일하게 재현할 수 없다.
3. 첫 실행 시 Whisper와 SentenceTransformer 모델 다운로드가 필요할 수 있다.
4. 외부 모델 및 패키지 버전 차이로 점수나 시각화 좌표가 완전히 일치하지 않을 수 있다.
5. 현재 저장소에는 이미 일부 산출물이 포함되어 있으므로, 완전 초기 상태 재현을 원하면 새 폴더에 저장소를 다시 복사한 뒤 실행하는 방식이 가장 명확하다.

---

## 14. 권장 재현 순서

가장 현실적인 재현 순서는 아래와 같다.

### 14.1 코드만으로 먼저 재현

```powershell
python run_audio_experiment_pipeline.py --regenerate-dataset --overwrite-audio --overwrite-stt
python html_experiment_app.py
```

### 14.2 이후 실데이터 재현

1. `data/audio/mp4/`에 MP4/WAV 배치
2. 아래 실행

```powershell
python run_real_mp4_pipeline.py --real-only --overwrite-audio --overwrite-stt --show-metrics
python html_experiment_app.py
```

이 순서가 현재 프로젝트의 기능을 가장 빠르게 검증하는 방법이다.
