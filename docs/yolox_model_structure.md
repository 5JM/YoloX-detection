# YOLOX 모델 구조

## 개요
- 전체 추론 파이프라인은 `CSPDarknet` 백본 → `YOLOPAFPN` 넥 → `YOLOXHead` 헤드로 구성되며, 래퍼 모듈 `YOLOX`가 학습·추론 인터페이스를 제공합니다.
- 이 프로젝트의 실험(`exps/smoke_ti.py`)은 YOLOX-Tiny 계열을 기반으로 깊이/너비, 흑백 입력, 활성화 함수를 변경해 경량 모델을 구성합니다.

## 구성 모듈 요약
| 구성 | 역할 | 핵심 구현 |
| --- | --- | --- |
| Backbone | 다양한 스케일의 특징 맵 생산 | `yolox/models/darknet.py:CSPDarknet` |
| Neck | Path Aggregation FPN으로 특징 융합 | `yolox/models/yolo_pafpn.py:YOLOPAFPN` |
| Head | 박스 회귀, 객체성, 클래스 분류 | `yolox/models/yolo_head.py:YOLOXHead` |
| Wrapper | 손실 계산 및 추론 결과 반환 | `yolox/models/yolox.py:YOLOX` |

## 데이터 흐름
1. 입력 이미지는 `stem → dark2 → dark3 → dark4 → dark5` 블록으로 구성된 `CSPDarknet`을 통과해 다중 해상도 특징(`dark3`, `dark4`, `dark5`)을 만듭니다.
2. `YOLOPAFPN`은 `dark5`에서 시작해 업샘플·컨캣·`CSPLayer` 과정을 거쳐 상향식 경로를 구축하고, 다시 다운샘플 경로를 통해 Path Aggregation을 완성합니다.
3. 세 해상도의 출력 `(P3, P4, P5)`는 `YOLOXHead`로 전달되어, 각 격자 위치별로 박스, 객체성, 클래스 로짓을 생산합니다.
4. 학습 모드에서는 `YOLOXHead.get_losses`가 SimOTA 기반 매칭을 수행하고 손실을 계산하며, 추론 모드에서는 디코딩된 박스(`decode_outputs`)를 반환합니다.

## Backbone: CSPDarknet
- `dep_mul`, `wid_mul`로 네트워크의 깊이·채널 폭을 조절합니다. 현재 실험에서는 `depth=0.33`, `width=0.25`로 설정됩니다.
- `stem`은 입력 채널 수에 맞춰 구성됩니다. `grayscale=True`일 때 `1→12→(64*width)` 경로로 시작해 흑백 영상 학습을 지원합니다.
- 각 스테이지는 `(Conv → CSPLayer)` 반복 구조이며, 마지막 `dark5`에는 `SPPBottleneck`을 추가해 수용영역을 확장합니다.
- `CSPLayer` 내부의 `Bottleneck`은 Shortcut을 유지한 채 채널을 분할·병합해 효율적인 표현 학습을 수행합니다.

## Neck: YOLOPAFPN
- 입력 특징 순서를 `(dark3, dark4, dark5)`로 받아, 가장 상위 해상도에서부터 상향식 특징 결합을 수행합니다.
- 주요 연산 흐름:
  - `lateral_conv0`로 `dark5` 채널을 축소 후 업샘플 → `torch.cat`으로 `dark4`와 결합 → `C3_p4`(`CSPLayer`)로 정련.
  - 같은 패턴으로 `dark3`까지 연결해 `pan_out2`를 생성.
  - 이후 `bu_conv2`, `bu_conv1`을 이용해 다시 낮은 해상도로 이동하며 Path Aggregation을 완성(`pan_out1`, `pan_out0`).
- `depthwise=False`이므로 `BaseConv` 기반 컨볼루션을 사용하며, 필요한 경우 `depthwise=True`로 경량화를 추가할 수 있도록 설계되어 있습니다.

## Head: YOLOXHead
- 각 피처맵마다 `stem(1x1 conv)` → `cls_convs`(3x3 conv 2회) / `reg_convs` 분기를 둬 분류·회귀 경로를 분리합니다.
- 최종 출력은 `[bbox(4), obj(1), class(num_classes)]`로 정렬되며, 학습 시에는 이 텐서를 기반으로 SimOTA 매칭을 수행합니다.
- 손실 계산에는 IoU 손실(`IOUloss`), BCE with logits, 필요 시 L1 손실이 포함되며, `initialize_biases`를 통해 초반 안정화를 돕습니다.
- 추론에서는 각 해상도 출력을 concatenate 후 그리드 오프셋과 stride를 적용하는 디코딩 과정을 거칩니다.

## Wrapper: YOLOX
- `YOLOX.forward`는 백본과 헤드를 호출하고, 학습 중이면 손실 딕셔너리를, 추론 중이면 디코딩된 박스/확률을 반환합니다.
- `visualize` 메서드는 어사인먼트 결과를 시각화(`yolox/utils/visualize_assign`)하는 기능을 노출합니다.

## 실험 스크립트(`exps/smoke_ti.py`) 특징
- 입력/테스트 크기: `320x320`, 배치 크기 128(기본) 기준 `basic_lr_per_img = 0.01 / (64*2)`.
- 데이터 경로: `datasets/{train, valid, test}` 폴더와 COCO 포맷 어노테이션(`train.json`, `valid.json`, `test.json`)을 기대합니다.
- Augmentation: Mosaic 비활성, MixUp 비활성, `flip_prob=0.5`, `translate=0.3`, `shear=2.0`, `mixup_scale=(1.0, 1.1)`.
- 훈련 전략: EMA 사용, 최대 700 epochs, `no_aug_epochs=50`, `eval_interval=1`로 매 에폭 검증.
- 클래스 수: 2개(흡연 여부). 필요 시 `self.num_classes`, `self.train_img_dir` 등을 수정해 새로운 도메인에 대응합니다.

위 구조를 기반으로 `tools/train.py`, `tools/eval.py`, `tools/demo.py` 등이 동일한 모듈을 재사용하며, 실험 파일에서 모든 하이퍼파라미터를 통제할 수 있습니다.
