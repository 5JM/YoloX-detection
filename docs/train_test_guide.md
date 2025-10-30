# YOLOX 학습·평가 가이드

## 준비 단계
- Python 3.8+와 CUDA 지원 PyTorch를 설치합니다. GPU 학습 시 `nvidia-smi`로 장치 상태를 확인합니다.
- 프로젝트 루트에서 `pip install -r requirements.txt`를 실행해 의존성을 맞춥니다.
- 로컬 CUDA/cuDNN 버전과 PyTorch 빌드가 호환되는지 확인합니다.

## 데이터셋 구성
1. 루트에 `datasets/`를 만들고 아래와 같이 이미지를 배치합니다.
   ```
   datasets/
     ├── train/
     ├── valid/
     └── test/
   ```
2. 각 세트에 대응하는 COCO 포맷 어노테이션(`train.json`, `valid.json`, `test.json`)을 `datasets/` 바로 아래에 둡니다.
3. `exps/smoke_ti.py`에서 `num_classes`, `train_img_dir`, `val_img_dir`, `test_img_dir`이 데이터와 맞는지 확인합니다.
4. 필요 시 `self.data_dir`를 직접 설정하거나 환경 변수 `YOLOX_DATADIR`를 사용해 데이터 경로를 재지정할 수 있습니다.

## 실험 스크립트 이해
- `exps/smoke_ti.py`는 YOLOX 실험 베이스를 상속하여 하이퍼파라미터, 데이터 전처리, 경로 등을 정의합니다.
- 주요 설정: 입력 크기 320, 클래스 2개, Mosaic/MixUp 비활성, ReLU 활성화, 흑백 입력(`grayscale=True`).
- 동일 구조의 실험을 추가하려면 파일을 복제 후 필요한 파라미터만 조정하고, 학습 실행 시 `-f` 옵션으로 새 스크립트를 지정합니다.

## 학습 실행
```bash
python tools/train.py \
  --project_name "smoke_detection" \
  -expn vaping_1004_3 \
  -n smoke \
  -f exps/smoke_ti.py \
  -b 128 \
  -d 1 \
  --fp16 \
  --cache ram
```
- `-b/--batch_size`: 전체 배치 크기. 실험 스크립트에서 `basic_lr_per_img`를 `0.01/batch`로 재설정하므로 CLI 배치 값과 일치시킵니다.
- `-d/--devices`: 사용할 GPU 수. 생략하면 `get_num_devices()`가 자동 결정합니다.
- `--resume`: `YOLOX_outputs/<expn>/latest_ckpt.pth`에서 학습을 이어갑니다. 필요 시 `--start_epoch`로 재개 에폭을 명시합니다.
- `--fp16`: AMP 기반 혼합 정밀 학습. 메모리와 속도를 절약합니다.
- `--cache {ram,disk}`: 데이터셋을 메모리/디스크에 캐싱해 I/O 병목을 줄입니다.
- 로깅 및 체크포인트는 `YOLOX_outputs/<experiment-name>/`에 저장되며, `--logger tensorboard|wandb`로 모니터링 도구를 선택할 수 있습니다.

## 평가 및 테스트
```bash
python tools/eval.py \
  -expn vaping_1004_3 \
  -f exps/smoke_ti.py \
  -c YOLOX_outputs/vaping_1004_3/best_ckpt.pth \
  --conf 0.4 \
  --nms 0.3 \
  --tsize 320
```
- `-c/--ckpt`: 평가할 체크포인트 경로. 기본값은 `YOLOX_outputs/<expn>/best_ckpt.pth`.
- `--conf`, `--nms`, `--tsize`: 추론 시의 confidence threshold, NMS 임계, 입력 크기를 제어합니다.
- `--fuse`: Conv+BN을 Fuse해 추론 속도를 높입니다.
- `--fp16`: Half precision 추론. `--trt`는 TensorRT 엔진을 사용할 때 지정합니다.
- 평가 로그, per-class AP/AR 요약은 `YOLOX_outputs/<expn>/val_log.txt`에 누적됩니다.

## 추가 유틸리티
- 단일 이미지/비디오 추론: `python tools/demo.py image -f exps/smoke_ti.py -c <ckpt> --path <입력 경로>`.
- ONNX 내보내기: `python tools/export_onnx.py -f exps/smoke_ti.py -c <ckpt> --output-name model.onnx`.
- TensorRT 변환: `python tools/trt.py -f exps/smoke_ti.py -c <ckpt>` 후 `tools/eval.py --trt`로 성능 측정.
- 체크포인트 관리: `YOLOX_outputs/<expn>/`에서 `best_ckpt.pth`, `latest_ckpt.pth`, 시드/로그 파일을 확인할 수 있습니다.

명령 실행 전 루트에서 `configure_module()`가 호출되어 경로 설정이 완료되므로, 항상 프로젝트 루트(`YoloX-detection/`)에서 스크립트를 실행하는 것을 권장합니다.
