# Audio-First Retrieval Experiment

긴 한국어 더미 텍스트를 만들고, 이를 TTS 오디오로 변환한 뒤, Whisper STT 결과와 원문 기준 검색 성능을 비교하는 실험 프로젝트입니다.

현재 기본 흐름은 다음과 같습니다.

`original_transcript -> TTS WAV 생성 -> Whisper STT -> stt_transcript -> 키워드 검색 / Dense 검색 / PCA / t-SNE / 군집 / 평가`

## 1. 환경 준비

### 가상환경 생성

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 패키지 설치

```powershell
pip install -r requirements-experiment.txt
```

### ffmpeg 확인

Whisper와 오디오 변환에 `ffmpeg`가 필요합니다.

```powershell
ffmpeg -version
```

## 2. 가장 자주 쓰는 실행 방법

### 전체 파이프라인 처음부터 다시 실행

더미데이터 100개 재생성, TTS, STT, 인덱스, 플롯, 군집, 평가까지 전부 다시 생성합니다.

```powershell
python run_audio_experiment_pipeline.py --regenerate-dataset --overwrite-audio --overwrite-stt
```

### 이미 만든 데이터 기준으로 다시 실행

기존 텍스트는 유지하고, 현재 메타데이터 기준으로 TTS/STT/검색 산출물만 다시 갱신합니다.

```powershell
python run_audio_experiment_pipeline.py --tts-provider edge --whisper-model base
```

### 앱 실행

```powershell
streamlit run experiment_app.py
```

앱이 열리면 검색 결과에 다음 정보가 함께 표시됩니다.

- 어떤 문서인지
- 몇 번째 줄인지
- 몇 번째 문장인지
- 가장 유사한 문장 본문

예시: `DOC-051의 17번째 줄 / 1번째 문장`

## 3. 현재 기준 추천 실행 순서

### 1) 전체 산출물 다시 만들기

```powershell
python run_audio_experiment_pipeline.py --regenerate-dataset --overwrite-audio --overwrite-stt
```

### 2) UI 확인

```powershell
streamlit run experiment_app.py
```

## 4. 부분 실행 명령어

### 더미데이터만 다시 생성

```powershell
python -m src.data.generate_dataset --total-items 100
```

### TTS만 다시 생성

```powershell
python -m src.audio.generate_tts_audio --provider edge --edge-voice ko-KR-SunHiNeural --overwrite
```

### STT만 다시 생성

```powershell
python -m src.stt.batch_transcribe --model-name base --overwrite
```

### Dense 임베딩 인덱스만 다시 생성

```powershell
python -m src.embedding.build_indices --text-sources stt_transcript original_transcript
```

### 평가만 다시 실행

```powershell
python -m src.evaluation.evaluate --text-sources stt_transcript original_transcript
```

### 텍스트 기반 파이프라인만 실행

오디오/TTS/STT 없이 텍스트 검색 실험만 돌릴 때 사용합니다.

```powershell
python run_experiment_pipeline.py --total-items 100
```

## 5. 플롯 관련 참고

현재 기본 플롯 재생성은 `PCA + t-SNE` 기준입니다.

- `UMAP`은 현재 환경에서 import/실행이 멈추는 문제가 있어 기본 재생성에서 제외했습니다.
- 앱에서는 실제로 존재하는 플롯만 노출되도록 되어 있습니다.
- 현재 UI에서 보이는 모드는 보통 다음 4개입니다.

```text
PCA 3D
PCA 2D
t-SNE 3D
t-SNE 2D
```

플롯만 다시 만들고 싶으면 아래처럼 실행합니다.

```powershell
python -m src.visualize.pca_plot --model-alias paraphrase-multilingual-MiniLM-L12-v2 --text-source stt_transcript
python -m src.visualize.pca_plot --model-alias paraphrase-multilingual-MiniLM-L12-v2 --text-source original_transcript
python -m src.visualize.pca_plot --model-alias multilingual-e5-base --text-source stt_transcript
python -m src.visualize.pca_plot --model-alias multilingual-e5-base --text-source original_transcript
```

## 6. 주요 산출물 위치

### 원문 텍스트

`data/raw`

### WAV 오디오

`data/audio/wav`

### Whisper 텍스트

`data/audio/stt_txt`

### 메타데이터

`data/metadata/dataset_metadata.csv`

### 임베딩

`artifacts/embeddings`

### 인덱스

`artifacts/indices`

### 플롯

`artifacts/plots`

### 군집 결과

`artifacts/clusters`

### 평가 결과

`artifacts/evaluation`

## 7. 현재 프로젝트 기본값

- 더미데이터 수: `100`
- TTS provider: `edge`
- Edge voice: `ko-KR-SunHiNeural`
- Whisper model: `base`
- 기본 검색 소스: `stt_transcript`
- 비교 소스: `stt_transcript`, `original_transcript`

## 8. 빠른 확인용 명령

현재 메타데이터가 있는지 확인:

```powershell
Get-Item data\metadata\dataset_metadata.csv
```

원문 파일 수 확인:

```powershell
(Get-ChildItem data\raw\*.txt).Count
```

WAV 파일 수 확인:

```powershell
(Get-ChildItem data\audio\wav\*.wav).Count
```

STT txt 파일 수 확인:

```powershell
(Get-ChildItem data\audio\stt_txt\*.txt).Count
```
