# vgg-16 모델을 이용한 나이 분류 (사용데이터셋: v_img)
import cv2
import time
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow import keras 
import random

# --- 1. 우리가 학습시킨 Keras 모델 및 클래스 이름 설정 ---
# 우리가 학습시킨 나이 분류 모델의 경로
OUR_AGE_MODEL_PATH = '/home/linux/final/1_Age-VGG161.keras' 

# 우리가 학습시킨 모델의 클래스 이름 (나이 범주 라벨)
# 이 라벨들은 당신의 모델이 예측하는 나이 범주와 순서가 정확히 일치해야 합니다.
# 모델의 마지막 Dense 레이어의 units 수와 이 리스트의 길이가 일치해야 합니다!
OUR_AGE_LABELS = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100+'] 

# --- 2. 우리가 학습시킨 Keras 모델 로드 ---
print(f"우리가 학습시킨 Keras 모델을 로드 중: {OUR_AGE_MODEL_PATH}...")
try:
    model = keras.models.load_model(OUR_AGE_MODEL_PATH, compile=False) 
    print(f"모델 '{OUR_AGE_MODEL_PATH}'이(가) 성공적으로 로드되었습니다.")
    model.summary() 
    
    if model.layers[-1].units != len(OUR_AGE_LABELS):
        print("\n[경고] 모델의 마지막 레이어 출력 유닛 수와 OUR_AGE_LABELS의 개수가 일치하지 않습니다!")
        print(f"모델 출력 유닛: {model.layers[-1].units}, OUR_AGE_LABELS 개수: {len(OUR_AGE_LABELS)}")
        print("모델이 나이 분류 모델이 아닐 수 있거나, OUR_AGE_LABELS가 잘못 정의되었을 수 있습니다.")
        
except Exception as e:
    print(f"오류: 모델 '{OUR_AGE_MODEL_PATH}' 로드 실패: {e}")
    print("모델 파일 경로와 파일 이름을 확인해주세요. 커스텀 객체(custom_objects)가 필요한지 확인해주세요.")
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

    processed_img = tf.keras.applications.vgg16.preprocess_input(img_array) 
    
    return processed_img

def post_process_our_model_output_age(predictions, class_labels):
    """
    우리가 학습시킨 나이 '분류' 모델의 출력에서 예측 나이 범주와 확률을 추출합니다.
    (모델이 분류 모델이고, Softmax 출력을 가정합니다.)
    """
    predicted_idx = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_idx] 
    predicted_confidence = predictions[0][predicted_idx]
    
    return predicted_label, predicted_confidence, predicted_idx

def get_age_category_idx(age, age_labels):
    """
    실제 나이(정수)에 해당하는 나이 범주의 인덱스를 반환합니다.
    이 함수는 `OUR_AGE_LABELS`의 정의와 일치해야 합니다.
    """
    for i, label in enumerate(age_labels):
        parts = label.split('-')
        if len(parts) == 2:
            try:
                min_age = int(parts[0])
                max_age_str = parts[1]
                if max_age_str.endswith('+'): 
                    max_age = float('inf') 
                else:
                    max_age = int(max_age_str)
                
                if min_age <= age <= max_age:
                    return i
            except ValueError:
                continue
        else:
            continue
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

# --- 6. 추론 실행 함수 (나이 분류 및 정확도 측정) ---
def run_ps(image_index, display=False):
    """
    지정된 이미지에 대해 Keras 기반 나이 분류 모델 추론을 수행합니다.
    정확도 측정을 위해 파일명에서 실제 나이를 파싱하여 범주와 비교합니다.
    """
    image_path = os.path.join(image_folder, original_images[image_index])
    input_image_cv2 = cv2.imread(image_path)
    
    if input_image_cv2 is None:
        print(f"경고: 이미지 '{image_path}'을(를) 읽을 수 없습니다. 건너뜀.")
        # 요청하신 출력 형식 유지
        print(f"  - N/A: N/A")
        print("Number of detected objects: 0")
        return 0, 0, None, None 

    # 파일명에서 실제 나이 정보 추출 (UTKFace 데이터셋 형식)
    filename_parts = original_images[image_index].split('_')
    
    true_age_int = -1
    true_age_idx = -1
    true_age_label = "N/A" 
    
    if len(filename_parts) > 0:
        try:
            true_age_int = int(filename_parts[0]) 
            true_age_idx = get_age_category_idx(true_age_int, OUR_AGE_LABELS)
            true_age_label = OUR_AGE_LABELS[true_age_idx] if true_age_idx != -1 else "Unknown Category"
        except (ValueError, IndexError):
            print(f"경고: 파일명 '{original_images[image_index]}'에서 유효한 나이 정보를 파싱할 수 없습니다.")
    else:
        print(f"경고: 파일명 '{original_images[image_index]}'이 UTKFace 형식에 맞지 않습니다. 나이 정보를 파싱할 수 없습니다.")


    # 2. 이미지 전처리
    processed_image = pre_process_for_our_model(input_image_cv2, model_input_size)

    # 3. 추론 실행
    predictions = model.predict(processed_image, verbose=0) 

    # 4. 출력 후처리
    predicted_label, predicted_confidence, predicted_idx = post_process_our_model_output_age(predictions, OUR_AGE_LABELS)

    # 예측이 실제 나이 범주와 일치하는지 확인
    is_correct = 1 if predicted_idx == true_age_idx and true_age_idx != -1 else 0

    if display:
        display_image = draw_predictions_on_image(input_image_cv2.copy(), predicted_label, predicted_confidence, true_age_label)
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
    print("Number of detected objects: 1") # 나이 분류는 단일 예측이므로 1로 고정
    if true_age_label != "N/A":
        print(f"  - Age Group: {true_age_label}")
       
    return 1, is_correct, predicted_label, true_age_label 

# --- 7. 성능 평가 수행 ---
print("\n--- 단일 이미지 Keras 나이 분류 모델 추론 테스트 ---")
if total_images > 0:
    _ , _ , _, _ = run_ps(0, display=True) 
else:
    print("테스트할 이미지가 없습니다.")

print("\n--- 전체 이미지 Keras 나이 분류 모델 추론 성능 평가 ---")
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

print("\n(CPU) 기반 VGG16 성능 평가 및 랜덤 이미지 분류 결과 표시 완료.") # 메시지 업데이트
print(f"총 소요 시간: {duration:.4f} 초")
print(f"나이 모델 추론 성능 (FPS): {fps_ps:.2f} FPS")
print(f"나이 모델 예측 정확도: {accuracy_ps:.2f}%")

