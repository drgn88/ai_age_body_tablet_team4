import cv2
import tensorflow as tf
import numpy as np
import time  # 시간 측정용

# 모델 경로
AGE_MODEL_PATH = "/home/linux/fmnist/2087_Age-VGG16.keras"
GENDER_MODEL_PATH = "/home/linux/fmnist/Final_10284_Gender-ResNet152.keras"

# 모델 로드
age_model = tf.keras.models.load_model(AGE_MODEL_PATH, compile=False)
gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH, compile=False)

# 설정
IMAGE_SIZE = 224
gender_mapping = ["Male", "Female"]

# 얼굴 탐지기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 상태 변수 초기화
show_age = False
age_display_start_time = 0
display_duration = 5
age_measurement_start_time = 0
age_to_display = None
gender_to_display = None
face_box = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # 첫 번째 얼굴만 사용
        (x, y, w, h) = faces[0]
        face_box = (x, y, w, h)

        face = frame[y:y+h, x:x+w]
        image = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        pred_age = age_model.predict(image, verbose=0)
        pred_gender = gender_model.predict(image, verbose=0)

        age = int(pred_age[0][0])
        gender = gender_mapping[int(np.round(pred_gender[0][0]))]

        gender_to_display = gender

        # 'a' 누른 후 3초 경과 시 나이 저장
        if show_age and age_to_display is None and current_time - age_measurement_start_time >= 3:
            age_to_display = age
            age_display_start_time = current_time

    # 라벨 생성 및 화면 표시
    if face_box is not None and gender_to_display is not None:
        (x, y, w, h) = face_box
        label = f"{gender_to_display}"

        if show_age and age_to_display is not None:
            if current_time - age_display_start_time < display_duration:
                label += f", Age: {age_to_display}"
            else:
                show_age = False
                age_to_display = None

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Age & Gender Prediction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        show_age = True
        age_measurement_start_time = time.time()
        age_to_display = None
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
