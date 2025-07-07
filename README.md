# TEAM: 조은지안조은지

## 팀원
- 조장: 정은지
- 팀원1: 장환
- 팀원2: 이은성
- 팀원3: 최현우

## Role
- 정은지
  - Team leader
- 이은성
  - Application developer(Raspberry pi)
- 장환
  - Validation
- 최현우
  - FPGA & On-device developer
  - 나이 및 성별 추론 모델 학습
    - 라즈베리파이 임베딩 및 성능 개선
  - Ultra96V2에 추론 모델 포팅
  - PYNQ OS importing
  - PS - PL에 따른 성능 비교
  - FPGA Vs 라즈베리파이 성능 비교


## 개발 일정

### <개발일정 기재 - Gantt Chart>

|                        |  7/1  |  7/2  |  7/3  |  7/4  |  7/5  |  7/6  |  7/7  |  7/8  |  7/9  | 7/10  |
| :--------------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 주제 선정              |   O   |       |       |       |       |       |       |       |       |       |
| 역할 분담              |   O   |       |       |       |       |       |       |       |       |       |
| 아이디어 제시          |   O   |   O   |       |       |       |       |       |       |       |       |
| 추론 모델 학습         |       |   O   |   O   |   O   |       |       |       |       |       |       |
| 추론 모델 개발 (PC)    |       |       |       |       |   O   |   O   |   O   |       |       |       |
| 라즈베리파이 임베딩    |       |       |       |       |       |       |   O   |   O   |       |       |
| FPGA 추론 모델 포팅    |       |       |       |       |   O   |   O   |   O   |   O   |       |       |
| FPGA-PS 추론 모델 제작 |       |       |       |       |       |       |       |   O   |   O   |   O   |
| 발표 자료 제작         |       |       |       |       |       |       |       |       |   O   |   O   |

### 0703~0704

#### 최현우

- 나이 학습 모델 개발및 추론 검증 병행
    - EffiecientNet B7모델 학습
    - EffiecientNet B7(Teacher) → B3(Student) 전이학습(경량화)
    - Gil Levi & Tal Hassner Model 레퍼런스 모델 참조
       - 라즈베리 파이 임베딩 후 성능 확인
       - 기존 학습 모델은 하나의 나이대만 추론
       - 레퍼런스 모델 사용시 다양한 나이대 추론 성공
    - 추론 모델 경량화 연구
       - Caffe → tflite 파일로 convert 수행
- FPGA이용 가속화 방법 연구
    - DPU IP를 사용하여 petalinux 이미지 bake 방안 모색
    - Vitis-AI 플랫폼
       - 학습 결과 파일 → .xmodel로 변환
       - 학습 결과에 적합한 DPU 가속기 생성지원
    - PYNQ OS
       - pynq-dpu를 이용하여 PYNQ OS에 내장된 DPU IP 사용가능
       - PYNQ-OS의 pynq-dpu와 VART를 사용하여 DPU를 이용한 추론

→ Vitis-AI 플랫폼을 이용한 .xmodel 우선 시도

### 0705-0706

#### 최현우

[07/05]

Ultra96 V2: Vitis-AI를 활용한 xmodel 생성 시도
- darknet yolov3의 weights 및 cfg를 caffe 형식으로 변환 시도
- Vitis-ai에서 지원하는 caffe 버전이 convert 스크립트의 concat 주변 레이어 구성 지원을 하지 않음
- tensorflow 형식으로 변환 시도
- yolov3-tiny모델의 weight와 cfg 파일의 불일치 문제
  - 해당 문제로 시간 지체
- yolov3의 weight와 cfg로 변환 시도
- frozen_graph.pb 파일 생성 완료
  - 최종 양자화 단계에서 .xmodel 파일 생성 오류
  - Segmentation fault 발생
  - calibration data 및 GPU 자원 off 시도
  - 그럼에도 해결 X

[07/06]

- PYNQ OS으로 해결 시도
- DPU 사용을 위한 .xmodel 생성 시도 실패
- PYNQ OS를 사용하여 Vitis-ai가 지원하는 pynq-dpu 사용 시도
- Ultra96v2에 DPU import 성공
- Yolov3를 이용한 이미지 객체 분류 성공
- 웹캠을 이용한 실시간 객체 분류 시도
  - PYNQ OS 모니터 출력과 웹캠 모니터 출력이 겹쳐 모니 송출 문제 발생
  - HW overlay에서 해당 DP 출력 IP를 찾아 웹캠 구동 시 xwindow 종료로 설정
  - 웹캠 구동은 성공했으나 카메라 송출 속도가 너무 느림
  - 이미지로 성능 비교하기로 방향 전환
- PS(ap)로만 사용했을 때와 PL(DPU)을 같이 사용했을 때 이미지 객체 분류에서 성능 비교 성공

### 0707

### 0708

### 0709

### 0710

## Trouble Shooting

### 이은성

### 장환

### 정은지

### 최현우
