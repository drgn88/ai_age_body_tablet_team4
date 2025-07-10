import cv2
import tensorflow as tf
import numpy as np
import time

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

# 이미지 불러오기 (JPG 경로 입력)
image_path = "/home/linux/fmnist/input/20_f_test_2.jpg"  # ← 원하는 이미지 파일로 수정
frame = cv2.imread(image_path)

if frame is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("얼굴을 찾을 수 없습니다.")
    exit()

# 감지된 얼굴들 중 가장 큰 얼굴 (넓이 기준) 하나만 선택
largest_face = None
max_area = 0

for (x, y, w, h) in faces:
    area = w * h
    if area > max_area:
        max_area = area
        largest_face = (x, y, w, h)

if largest_face is None:
    print("가장 큰 얼굴을 찾을 수 없습니다 (이런 경우는 거의 없음).")
    exit()

# 가장 큰 얼굴 정보 할당
x, y, w, h = largest_face

print("얼굴을 감지했습니다. 3초 후 나이와 성별을 예측합니다...")
time.sleep(0) # time.sleep(0)은 사실상 sleep하지 않으므로 필요에 따라 0이 아닌 값으로 변경하세요.

# --- 사용자 정의 설정 변수 ---
BOX_THICKNESS = 5
FONT_SCALE = 3
TEXT_THICKNESS = 4
# --- --- ---

# 이제 가장 큰 얼굴 하나에 대해서만 처리합니다.
face = frame[y:y+h, x:x+w]
image = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
image = image / 255.0
image = np.expand_dims(image, axis=0)

# 나이 중앙값 추정
age_predictions = [age_model.predict(image, verbose=0)[0][0] for _ in range(15)]
raw_age = np.median(age_predictions)

# --- 변경된 나이 보정 로직 시작 ---
adjusted_age = raw_age # 기본값으로 raw_age 설정

if 20 <= raw_age < 30:
    adjusted_age = raw_age - 7
elif 30 <= raw_age < 40:
    adjusted_age = raw_age - 4
elif 40 <= raw_age < 50:
    adjusted_age = raw_age + 5
elif 50 <= raw_age < 60:
    adjusted_age = raw_age + 8
elif 60 <= raw_age < 70:
    adjusted_age = raw_age + 13
elif 70 <= raw_age < 80: # 70대
    adjusted_age = raw_age + 18
# 80세 이상이나 20세 미만 등 다른 범위에 대한 처리가 필요하다면 여기에 추가할 수 있습니다.
# 예를 들어, 80세 이상은 80으로 클리핑하거나 특정 값을 더할 수 있습니다.

# 최종 나이를 0에서 80 사이로 클리핑
age = int(np.clip(adjusted_age, 0, 80))
# --- 변경된 나이 보정 로직 끝 ---

# 성별 예측
pred_gender = gender_model.predict(image, verbose=0)
gender = gender_mapping[int(np.round(pred_gender[0][0]))]

label = f"{gender}, Age: {age}"

# 박스 그리기
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), BOX_THICKNESS)

# 텍스트 그리기
cv2.putText(frame, label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), TEXT_THICKNESS)

# 결과 저장
output_path = "/home/linux/fmnist/output/output.jpg"
cv2.imwrite(output_path, frame)
print(f"결과 이미지를 저장했습니다: {output_path}")
#
#import cv2
#import tensorflow as tf
#import numpy as np
#import time
#
## 모델 경로
#AGE_MODEL_PATH = "/home/linux/fmnist/2087_Age-VGG16.keras"
#GENDER_MODEL_PATH = "/home/linux/fmnist/Final_10284_Gender-ResNet152.keras"
#
## 모델 로드
#age_model = tf.keras.models.load_model(AGE_MODEL_PATH, compile=False)
#gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH, compile=False)
#
## 설정
#IMAGE_SIZE = 224
#gender_mapping = ["Male", "Female"]
#
## 얼굴 탐지기 로드
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
## 이미지 불러오기 (JPG 경로 입력)
#image_path = "/home/linux/fmnist/input/50/50_0_w1.png"  # ← 원하는 이미지 파일로 수정
#frame = cv2.imread(image_path)
#
#if frame is None:
#    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
#
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#if len(faces) == 0:
#    print("얼굴을 찾을 수 없습니다.")
#    exit()
#
## --- 변경된 부분 시작 ---
## 감지된 얼굴들 중 가장 큰 얼굴 (넓이 기준) 하나만 선택
#largest_face = None
#max_area = 0
#
#for (x, y, w, h) in faces:
#    area = w * h
#    if area > max_area:
#        max_area = area
#        largest_face = (x, y, w, h)
#
#if largest_face is None: # 혹시 모를 경우를 대비
#    print("가장 큰 얼굴을 찾을 수 없습니다 (이런 경우는 거의 없음).")
#    exit()
#
## 가장 큰 얼굴 정보 할당
#x, y, w, h = largest_face
## --- 변경된 부분 끝 ---
#
#print("얼굴을 감지했습니다. 3초 후 나이와 성별을 예측합니다...")
#time.sleep(0)
#
## --- 사용자 정의 설정 변수 (이전과 동일) ---
#BOX_THICKNESS = 2
#FONT_SCALE = 0.8
#TEXT_THICKNESS = 1
## --- --- ---
#
## 이제 가장 큰 얼굴 하나에 대해서만 처리합니다.
#face = frame[y:y+h, x:x+w]
#image = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
#image = image / 255.0
#image = np.expand_dims(image, axis=0)
#
## 나이 중앙값 추정
#age_predictions = [age_model.predict(image, verbose=0)[0][0] for _ in range(15)]
#raw_age = np.median(age_predictions)
#age = int(np.clip(raw_age - 7 + 12, 0, 80))  # 보정값 7 적용 후 클리핑
#
## 성별 예측
#pred_gender = gender_model.predict(image, verbose=0)
#gender = gender_mapping[int(np.round(pred_gender[0][0]))]
#
#label = f"{gender}, Age: {age}"
#
## 박스 그리기
#cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), BOX_THICKNESS)
#
## 텍스트 그리기
#cv2.putText(frame, label, (x, y-10),
#            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), TEXT_THICKNESS)
#
## 결과 저장
#output_path = "/home/linux/fmnist/output/output.jpg"
#cv2.imwrite(output_path, frame)
#print(f"결과 이미지를 저장했습니다: {output_path}")
#*/