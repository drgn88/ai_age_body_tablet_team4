import cv2
import time
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow import keras
# ResNet50V2 전처리를 위해 필요한 모듈 임포트
# 성별 모델이 ResNet152V2 기반이라고 했으므로, 해당 전처리를 사용합니다.
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input

# --- 1. 우리가 학습시킨 Keras 모델 및 클래스 이름 설정 ---
# 우리가 학습시킨 성별 분류 모델의 경로
# *** 이 경로는 당신의 실제 성별 모델 경로로 변경해야 합니다! ***
OUR_GENDER_MODEL_PATH = '/home/linux/fmnist/Gender-VGG16.keras' # 예시 경로입니다. 실제 경로로 변경하세요.

# 우리가 학습시킨 모델의 클래스 이름 (성별 라벨)
# 0은 Male, 1은 Female에 해당해야 합니다.
OUR_GENDER_LABELS = ["Male", "Female"]

# --- 2. 우리가 학습시킨 Keras 모델 로드 ---
print(f"우리가 학습시킨 Keras 모델을 로드 중: {OUR_GENDER_MODEL_PATH}...")
try:
    # compile=False로 로드하여, 모델이 학습 시 사용한 컴파일 설정을 따르도록 합니다.
    model = keras.models.load_model(OUR_GENDER_MODEL_PATH, compile=False)
    print(f"모델 '{OUR_GENDER_MODEL_PATH}'이(가) 성공적으로 로드되었습니다.")
    model.summary()
except Exception as e:
    print(f"오류: 모델 '{OUR_GENDER_MODEL_PATH}' 로드 실패: {e}")
    raise SystemExit("모델 로드 실패.")

# --- 3. 모델 관련 설정 ---
# 우리가 모델을 학습시킬 때 사용한 입력 크기를 확인하고 설정합니다.
# 이 값은 당신의 성별 모델 학습 시 설정된 입력 크기와 일치해야 합니다.
model_input_size = (224, 224)

# --- 4. 평가 관련 유틸리티 함수 정의 ---
def pre_process_for_our_model(image_cv2, input_size):
    """
    우리가 학습시킨 성별 모델에 적합한 이미지 전처리 (리사이징, 배열 변환, 스케일링 등).
    ResNet 계열 모델의 표준 전처리 방식을 여기에 적용합니다.
    """
    # 1. BGR 이미지를 RGB로 변환 (cv2.imread는 BGR을 반환, ResNet 계열은 RGB를 기대)
    img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    # 2. 이미지 리사이징
    img_resized = cv2.resize(img_rgb, input_size)

    # 3. Keras 이미지 배열로 변환 (높이, 너비, 채널)
    img_array = image.img_to_array(img_resized)

    # 4. 배치 차원 추가 (모델 입력 형태에 맞게)
    img_array = np.expand_dims(img_array, axis=0)

    # *** 핵심: ResNet 계열의 표준 전처리 함수를 사용합니다. ***
    # 이 함수는 ImageNet 데이터셋의 평균 및 표준 편차를 기반으로 픽셀 값을 정규화합니다.
    processed_img = resnet_preprocess_input(img_array)

    return processed_img

def post_process_our_model_output_gender(predictions, class_labels):
    """
    우리가 학습시킨 성별 '분류' 모델의 출력에서 예측 성별과 확률을 추출합니다.
    이진 분류(Sigmoid) 출력을 가정합니다.
    """
    # predictions는 (1, 1) 형태의 배열일 것입니다 (Sigmoid 출력).
    # 0.5를 기준으로 반올림하여 0 또는 1로 분류합니다.
    predicted_raw_value = predictions[0][0]
    predicted_idx = int(np.round(predicted_raw_value)) # 0 또는 1

    predicted_label = class_labels[predicted_idx] # 예: 'Male' 또는 'Female'
    # 이진 분류에서는 예측값 자체가 신뢰도로 볼 수 있습니다.
    # 0.5에서 얼마나 멀리 떨어져 있는지로 신뢰도를 간접적으로 판단할 수도 있습니다.
    # 여기서는 예측된 클래스의 '확률'을 표시합니다.
    # 만약 sigmoid 출력이라면, predicted_raw_value가 곧 1일 확률입니다.
    predicted_confidence = predicted_raw_value if predicted_idx == 1 else (1 - predicted_raw_value)

    return predicted_label, predicted_confidence, predicted_idx

def get_gender_category_idx(gender_int, gender_labels):
    """
    실제 성별(정수 0 또는 1)에 해당하는 인덱스를 반환합니다.
    """
    # UTKFace 데이터셋에서 0은 Male, 1은 Female이므로, 인덱스와 직접 매칭됩니다.
    if gender_int == 0 or gender_int == 1:
        return gender_int
    return -1 # 유효하지 않은 성별 값

def draw_predictions_on_image(image_cv2, predicted_label, predicted_confidence, true_label=None):
    """
    이미지에 모델 분류 결과를 텍스트로 그립니다.
    """
    result_image = image_cv2.copy()
    text = f"{predicted_label} ({predicted_confidence:.2f})"
    if true_label is not None:
        text += f" | True: {true_label}"

    cv2.putText(result_image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return result_image

# --- 5. 테스트 이미지 경로 설정 ---
image_folder = '/home/linux/fmnist/UTKFace' # UTKFace 폴더 경로

if not os.path.exists(image_folder):
    print(f"오류: 이미지 폴더 '{image_folder}'을(를) 찾을 수 없습니다.")
    raise SystemExit("이미지 폴더 누락.")

# UTKFace 이미지 파일명은 [age]_[gender]_[race]_[date&time].jpg 형식입니다.
original_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
original_images.sort() # 순서 보장
total_images = len(original_images)

if total_images == 0:
    print(f"오류: '{image_folder}' 폴더에 이미지 파일이 없습니다. 이미지를 추가해주세요.")
    raise SystemExit("테스트 이미지 없음.")

print(f"'{image_folder}'에서 {total_images}개의 테스트 이미지를 찾았습니다.")

# --- 6. PS (CPU/GPU) 추론 실행 함수 (성별 예측 및 정확도 측정) ---
def run_ps_gender_model(image_index, display=False): # TTA 옵션 제거
    """
    지정된 이미지에 대해 PS (CPU/GPU) 기반 성별 분류 모델 추론을 수행합니다.
    정확도 측정을 위해 파일명에서 실제 성별을 파싱하여 범주와 비교합니다.
    """
    image_path = os.path.join(image_folder, original_images[image_index])
    input_image_cv2 = cv2.imread(image_path) # BGR 형식으로 이미지 로드

    if input_image_cv2 is None:
        print(f"경고: 이미지 '{image_path}'을(를) 읽을 수 없습니다. 건너뜀.")
        return 0, 0, None, None

    # 파일명에서 실제 성별 정보 추출 (UTKFace 데이터셋 형식)
    filename_parts = original_images[image_index].split('_')

    if len(filename_parts) < 2: # 성별 정보는 두 번째 요소
        print(f"경고: 파일명 '{original_images[image_index]}'이 UTKFace 형식에 맞지 않습니다. 성별 정보를 파싱할 수 없습니다.")
        return 0, 0, None, None

    try:
        true_gender_int = int(filename_parts[1]) # 두 번째 요소가 실제 성별 (0 또는 1)
        true_gender_idx = get_gender_category_idx(true_gender_int, OUR_GENDER_LABELS)
        true_gender_label = OUR_GENDER_LABELS[true_gender_idx] if true_gender_idx != -1 else "Unknown"
    except (ValueError, IndexError):
        print(f"경고: 파일명 '{original_images[image_index]}'에서 유효한 성별 정보를 파싱할 수 없습니다.")
        return 0, 0, None, None

    # 이미지 전처리 (BGR -> RGB 변환, 리사이징, 정규화)
    processed_image = pre_process_for_our_model(input_image_cv2, model_input_size)

    # 추론 실행
    predictions = model.predict(processed_image, verbose=0)

    # 최종 예측 후처리
    predicted_label, predicted_confidence, predicted_idx = \
        post_process_our_model_output_gender(predictions, OUR_GENDER_LABELS)

    # 예측이 실제 성별과 일치하는지 확인
    is_correct = 1 if predicted_idx == true_gender_idx else 0

    if display:
        display_image = draw_predictions_on_image(input_image_cv2.copy(), predicted_label, predicted_confidence, true_gender_label)
        if display_image is not None:
            max_display_width = 800
            if display_image.shape[1] > max_display_width:
                scale_factor = max_display_width / display_image.shape[1]
                display_image = cv2.resize(display_image, (max_display_width, int(display_image.shape[0] * scale_factor)))

            cv2.imshow(f"Prediction for {original_images[image_index]}", display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("경고: 이미지를 표시할 수 없습니다 (display_image가 None).")

        print(f"Image: {original_images[image_index]}, Predicted: {predicted_label} ({predicted_confidence:.2f}), True: {true_gender_label}")
    return 1, is_correct, predicted_label, true_gender_label

# --- 7. 성능 평가 수행 ---
print("\n--- 단일 이미지 PS (CPU/GPU) 추론 테스트 (성별 모델) ---")
if total_images > 0:
    # 첫 번째 이미지를 테스트 (인덱스 0)
    print("단일 이미지 테스트:")
    _, _, _, _ = run_ps_gender_model(0, display=True)
else:
    print("테스트할 이미지가 없습니다.")

print("\n--- 전체 이미지 PS (CPU/GPU) 추론 성능 평가 (성별 모델) ---")
total_successful_inferences = 0
total_correct_predictions = 0

print("전체 이미지 평가 시작...")
time1 = time.time()
num_total_predictions = 0 # 실제 모델 추론 횟수를 계산하기 위함
for i in range(total_images):
    success, is_correct, _, _ = run_ps_gender_model(i, display=False)

    if success:
        total_successful_inferences += 1
        total_correct_predictions += is_correct
        num_total_predictions += 1 # TTA 제거했으므로, 원본 이미지 당 1회 추론

time2 = time.time()

duration = time2 - time1
fps_ps = 0.0
accuracy_ps = 0.0

if duration > 0 and total_successful_inferences > 0:
    fps_ps = total_successful_inferences / duration
    accuracy_ps = (total_correct_predictions / total_successful_inferences) * 100

print("\n(CPU) 기반 VGG16 성능 평가 및 랜덤 이미지 분류 결과 표시 완료.") # 메시지 업데이트
print(f"총 소요 시간: {duration:.4f} 초")
print(f"나이 모델 추론 성능 (FPS): {fps_ps:.2f} FPS")
print(f"나이 모델 예측 정확도: {accuracy_ps:.2f}%")
