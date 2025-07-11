# 🚀Trouble Shooting - Integrated System

## 문제 인지
- 웹캠 프레임에서 얼굴 검출과 추론을 수행하면서 영상이 버벅이고 느려지는 현상이 발생

| 문제 |
| :---: |
|![alt text](../video/problem1.gif)|

## 원인 파악
- 매 프레임 마다 얼굴 검출 및 추론
    - **CPU 과도 사용**
    - **웹캠 영상 끊기거나 응답 지연 발생**

## 문제 해결
- 예측 빈도를 낮추고 FPS를 20으로 고정시켜 불필요한 추론 연산을 감소시켜 리소스를 감소시키는 방안 고안

```py
# FPS 20으로 고정
TARGET_FPS = 20, 20
FRAME_MS = int(1000 / TARGET_FPS)

# 예측 빈도 설정
PREDICTION_INTERVAL_FRAMES = 50 
```

- CPU 사용률 감소, 상대적으로 웹캠 영상 부드럽게 출력
- ❗ 정확도의 큰 차이는 없음

