# ResNet-50 모델을 이용한 성별분류 (사용 데이터셋 : v_img)
#=====================================================================
import cv2
import time이
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow import keras 
import random

# --- 1. 우리가 학습시킨 Keras 모델 및 클래스 이름 설정 (성별 분류용) ---
# 우리가 학습시킨 성별 분류 모델의 경로
# 이 경로는 당신의 실제 성별 모델 경로로 변경해야 합니다!

# 우리가 학습시킨 모델의 클래스 이름 (성별 라벨)
# 이 라벨들은 당신의 모델이 예측하는 성별 범주와 순서가 정확히 일치해야 합니다.
# UTKFace 데이터셋의 성별: 0은 남성 (Male), 1은 여성 (Female)
# 모델이 Male에 대해 낮은 확률 (0에 가까움), Female에 대해 높은 확률 (1에 가까움)을 출력한다면 Male이 0번 인덱스, Female이 1번 인덱스여야 합니다.
OUR_GENDER_LABELS = ['Male', 'Female'] 

# --- 2. ResNet50 모델 로드 (TensorFlow Keras) ---
print("ResNet50 모델을 TensorFlow Keras로 로드 중...")
try:
    # ImageNet 가중치와 함께 ResNet50 모델을 로드합니다.
    # include_top=True는 분류를 위한 마지막 Dense 레이어를 포함함을 의미합니다.
    model = ResNet50(weights='imagenet')
    print("ResNet50 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"오류: ResNet50 모델 로드 실패: {e}")
    print("인터넷 연결을 확인하여 ImageNet 가중치를 다운로드할 수 있는지 확인해주세요.")
    raise SystemExit("모델 로드 실패.")

# --- 3. 모델 관련 설정 ---
model_input_size = (224, 224) 

# --- 4. 평가 관련 유틸리티 함수 정의 ---
def pre_process_for_our_model(image_cv2, input_size):
    """
    우리가 학습시킨 모델에 적합한 이미지 전처리 (리사이징, 배열 변환, 스케일링 등).
    모델 학습 시 사용한 전처리 방식을 여기에 적용해야 합니다.
    """
    img_resized = cv2.resize(image_cv2, input_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) 

    # --- 변경된 부분: ResNet50의 preprocess_input 사용 ---
    processed_img = tf.keras.applications.resnet50.preprocess_input(img_array) 
    
    return processed_img

def post_process_our_model_output_gender(predictions, class_labels):
    """
    우리가 학습시킨 성별 '분류' 모델의 출력에서 예측 성별 라벨과 확률을 추출합니다.
    (모델이 이진 분류 모델이고, Sigmoid 출력을 가정합니다.)
    """
    prob_female = predictions[0][0] 
    
    if prob_female >= 0.5:
        predicted_idx = 1 # 'Female' (OUR_GENDER_LABELS의 인덱스 1)
        predicted_label = class_labels[predicted_idx]
        predicted_confidence = prob_female
    else:
        predicted_idx = 0 # 'Male' (OUR_GENDER_LABELS의 인덱스 0)
        predicted_label = class_labels[predicted_idx]
        predicted_confidence = 1 - prob_female 

    return predicted_label, predicted_confidence, predicted_idx

def get_gender_category_idx(gender_int, gender_labels):
    """
    실제 성별(정수: 0=Male, 1=Female)에 해당하는 성별 라벨의 인덱스를 반환합니다.
    이 함수는 `OUR_GENDER_LABELS`의 정의와 UTKFace 데이터셋의 성별 인코딩과 일치해야 합니다.
    """
    if gender_int == 0:
        return gender_labels.index('Male') if 'Male' in gender_labels else -1
    elif gender_int == 1:
        return gender_labels.index('Female') if 'Female' in gender_labels else -1
    return -1 

def draw_predictions_on_image(image_cv2, predicted_label, predicted_confidence, true_label=None):
    """
    이미지에 모델 분류 결과를 텍스트로 그립니다.
    """
    result_image = image_cv2.copy()
    text = f"Predicted: {predicted_label} ({predicted_confidence:.2f})"
    if true_label is not None:
        text += f" | True: {true_label}"
    
    cv2.putText(result_image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return result_image

# --- 5. 테스트 이미지 경로 설정 (UTKFace 폴더 형식) ---
image_folder = '/home/linux/final/v_img' 

if not os.path.exists(image_folder):
    print(f"오류: 이미지 폴더 '{image_folder}'을(를) 찾을 수 없습니다.")
    raise SystemExit("이미지 폴더 누락.")

original_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
original_images.sort() 
total_images = len(original_images)

if total_images == 0:
    print(f"오류: '{image_folder}' 폴더에 이미지 파일이 없습니다. 이미지를 추가해주세요.")
    raise SystemExit("테스트 이미지 없음.")

print(f"'{image_folder}'에서 {total_images}개의 테스트 이미지를 찾았습니다.")

# --- 6. 추론 실행 함수 (성별 분류 및 정확도 측정) ---
def run_ps(image_index, display=False):
    """
    지정된 이미지에 대해 Keras 기반 성별 분류 모델 추론을 수행합니다.
    정확도 측정을 위해 파일명에서 실제 성별을 파싱하여 범주와 비교합니다.
    """
    image_path = os.path.join(image_folder, original_images[image_index])
    input_image_cv2 = cv2.imread(image_path)
    
    if input_image_cv2 is None:
        print(f"경고: 이미지 '{image_path}'을(를) 읽을 수 없습니다. 건너뜀.")
        print(f"  - N/A: N/A")
        print("Number of detected objects: 0")
        return 0, 0, None, None 

    # 파일명에서 실제 성별 정보 추출 (UTKFace 데이터셋 형식)
    filename_parts = original_images[image_index].split('_')
    
    true_gender_int = -1
    true_gender_idx = -1
    true_gender_label = "N/A" 
    
    if len(filename_parts) > 1:
        try:
            true_gender_int = int(filename_parts[1]) 
            true_gender_idx = get_gender_category_idx(true_gender_int, OUR_GENDER_LABELS)
            true_gender_label = OUR_GENDER_LABELS[true_gender_idx] if true_gender_idx != -1 else "Unknown Category"
        except (ValueError, IndexError):
            print(f"경고: 파일명 '{original_images[image_index]}'에서 유효한 성별 정보를 파싱할 수 없습니다.")
    else:
        print(f"경고: 파일명 '{original_images[image_index]}'이 UTKFace 형식에 맞지 않습니다. 성별 정보를 파싱할 수 없습니다.")


    # 2. 이미지 전처리
    processed_image = pre_process_for_our_model(input_image_cv2, model_input_size)

    # 3. 추론 실행
    predictions = model.predict(processed_image, verbose=0) 

    # 4. 출력 후처리
    predicted_label, predicted_confidence, predicted_idx = post_process_our_model_output_gender(predictions, OUR_GENDER_LABELS)

    # 예측이 실제 성별 범주와 일치하는지 확인
    is_correct = 1 if predicted_idx == true_gender_idx and true_gender_idx != -1 else 0

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
            
    # 이미지 추론 시마다 CLI에 결과 출력
    print(f"  - Gender: {predicted_label} ({predicted_confidence:.2f})")
    print("Number of detected objects: 1") 
    if true_gender_label != "N/A":
        print(f"  - True Gender: {true_gender_label}")

    
    return 1, is_correct, predicted_label, true_gender_label 

# --- 7. 성능 평가 수행 ---
print("\n--- 단일 이미지 Keras 성별 분류 모델 추론 테스트 ---")
if total_images > 0:
    _ , _ , _, _ = run_ps(0, display=True) 
else:
    print("테스트할 이미지가 없습니다.")

print("\n--- 전체 이미지 Keras 성별 분류 모델 추론 성능 평가 ---")
total_successful_inferences = 0 
total_correct_predictions = 0   

time1 = time.time()
for i in range(total_images):
    success, is_correct, _, _ = run_ps(i, display=False) 
    if success: 
        total_successful_inferences += 1
        total_correct_predictions += is_correct
time2 = time.time()

duration = time2 - time1
fps_ps = 0.0
accuracy_ps = 0.0

if duration > 0 and total_successful_inferences > 0:
    fps_ps = total_successful_inferences / duration
    accuracy_ps = (total_correct_predictions / total_successful_inferences) * 100

print("\nPS(CPU) 기반 성능 평가 및 랜덤 이미지 분류 결과 표시 완료.")
print(f"총 소요 시간: {duration:.4f} 초")
print("Performance PS(CPU): {} FPS".format(fps_ps))
print(f"성별 모델 예측 정확도: {accuracy_ps:.2f}%")