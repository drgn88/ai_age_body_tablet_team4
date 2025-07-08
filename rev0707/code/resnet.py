# 동물 사진 분류 (사용 데이터셋 : img)
import cv2
import time
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import random # --- 추가: random 모듈 임포트 ---

# --- 1. ResNet50 모델 및 클래스 이름 설정 (ImageNet) ---
# ResNet50은 ImageNet 데이터셋으로 학습되었으므로 별도의 .names 파일이 필요하지 않습니다.
# decode_predictions 함수가 ImageNet 클래스 이름을 제공합니다.

# --- 2. ResNet50 모델 로드 (TensorFlow Keras) ---
print("ResNet50 모델을 TensorFlow Keras로 로드 중...")
try:
    # ImageNet 가중치와 함께 ResNet50 모델을 로드합니다.
    # include_top=True는 분류를 위한 마지막 Dense 레이어를 포함함을 의미합니다.
    model = ResNet50(weights='imagenet')
    print("ResNet50 모델이 성공적으로 로드되었습니다.류")
except Exception as e:
    print(f"오류: ResNet50 모델 로드 실패: {e}")
    print("인터넷 연결을 확인하여 ImageNet 가중치를 다운로드할 수 있는지 확인해주세요.")
    raise SystemExit("모델 로드 실패.")

# --- 3. 클래스 이름 (ImageNet) ---
# decode_predictions 함수가 내부적으로 클래스 이름을 처리하므로 명시적인 로드는 필요 없습니다.
# 여기서는 편의상 주석으로 처리합니다.
# class_names = [...] # ImageNet 1000개 클래스 이름

# --- 4. ResNet50 관련 설정 ---
# ResNet50 모델의 입력 크기는 224x224입니다.
model_input_size = (224, 224)

# --- 5. 평가 관련 유틸리티 함수 정의 ---

def pre_process_for_ps(image_cv2, input_size):
    """
    TensorFlow ResNet50에 적합한 이미지 전처리 (리사이징, 배열 변환, 스케일링 등).
    cv2 이미지를 Keras 모델 입력 형식으로 변환합니다.
    """
    # OpenCV 이미지를 PIL Image 형식으로 변환 (TensorFlow Image Preprocessing에 더 적합)
    # 또는 직접 NumPy 배열로 조작 가능
    # ResNet50의 preprocess_input은 [-1, 1] 범위로 픽셀을 스케일링하고 BGR->RGB 변환을 수행합니다.
    img_resized = cv2.resize(image_cv2, input_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) # 배치 차원 추가
    processed_img = preprocess_input(img_array)
    return processed_img

def post_process_resnet_output(predictions, top_n=5):
    """
    ResNet50 출력 (확률)에서 상위 N개 클래스를 추출합니다.
    """
    # decode_predictions 함수는 (클래스 ID, 클래스 이름, 확률) 튜플 리스트를 반환합니다.
    decoded_predictions = decode_predictions(predictions, top=top_n)[0]
    return decoded_predictions # 예: [('n02123597', 'Siamese_cat', 0.92...), ...]

def draw_predictions_on_image(image_cv2, predictions_list):
    """
    이미지에 ResNet50 분류 결과를 텍스트로 그립니다.
    """
    result_image = image_cv2.copy()
    y_offset = 30
    for i, (imagenet_id, label, score) in enumerate(predictions_list):
        text = f"{label}: {score:.2f}"
        cv2.putText(result_image, text, (10, y_offset + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return result_image

# --- 6. 테스트 이미지 경로 설정 (DPU 코드와 동일하게) ---
image_folder = 'img' # 당신의 이미지 폴더 경로
if not os.path.exists(image_folder):
    print(f"오류: 이미지 폴더 '{image_folder}'을(를) 찾을 수 없습니다.")
    print("테스트 이미지를 포함하는 'img' 폴더를 생성해주세요.")
    raise SystemExit("이미지 폴더 누락.")

original_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
original_images.sort() # 순서 보장
total_images = len(original_images)

if total_images == 0:
    print(f"오류: '{image_folder}' 폴더에 이미지 파일이 없습니다. 이미지를 추가해주세요.")
    raise SystemExit("테스트 이미지 없음.")

print(f"'{image_folder}'에서 {total_images}개의 테스트 이미지를 찾았습니다.")

# --- 7. PS (CPU/GPU) 추론 실행 함수 (DPU 코드의 run 함수와 유사하게 재구성) ---
def run_ps(image_index, display=False):
    """
    지정된 이미지에 대해 PS(CPU) 기반 ResNet50 추론을 수행합니다.
    """
    # 1. 이미지 로드
    image_path = os.path.join(image_folder, original_images[image_index])
    input_image_cv2 = cv2.imread(image_path)
    if input_image_cv2 is None:
        print(f"경고: 이미지 '{image_path}'을(를) 읽을 수 없습니다. 건너뜀.")
        print("Number of detected objects: 0") # 분류 결과는 객체 수로 표현하기 어렵지만, 일관성을 위해
        return 0, []

    # 2. 이미지 전처리
    processed_image = pre_process_for_ps(input_image_cv2, model_input_size)

    # 3. 추론 실행
    predictions = model.predict(processed_image, verbose=0) # verbose=0으로 출력 억제

    # 4. 출력 후처리 (상위 5개 예측)
    top_predictions = post_process_resnet_output(predictions, top_n=1)

    # 분류 결과는 객체 탐지처럼 "객체 수"로 명확히 표현되지 않지만,
    # YOLO 코드와의 일관성을 위해 상위 예측 개수를 반환합니다.
    num_predictions = len(top_predictions)

    if display:
        display_image = draw_predictions_on_image(input_image_cv2.copy(), top_predictions)
        if display_image is not None:
            max_display_width = 800
            if display_image.shape[1] > max_display_width:
                scale_factor = max_display_width / display_image.shape[1]
                display_image = cv2.resize(display_image, (max_display_width, int(display_image.shape[0] * scale_factor)))
            
            # IPython.display 대신 cv2.imshow를 사용합니다.
            # 주피터 노트북이나 코랩에서는 이 부분을 주석 처리하고 IPython.display 사용 가능
            cv2.imshow(f"Classification Result: {original_images[image_index]}", display_image)
            cv2.waitKey(0) # 아무 키나 누를 때까지 창을 유지
            cv2.destroyAllWindows() # 창 닫기
            
    print(f"Number of detected objects: {num_predictions}")
    for _, label, score in top_predictions:
        print(f"  - {label}: {score:.2f}")
    return num_predictions, top_predictions

# --- 추가: 랜덤 이미지 분류 결과만 표시 ---
if total_images > 0:
    # --- 변경된 부분: 랜덤 인덱스 선택 ---
    random_index = random.randint(0, total_images - 1)
    # --- 변경된 부분 끝 ---
    print(f"\n--- 랜덤 이미지 ({original_images[random_index]}) 분류 결과 표시 ---")
    _ , _ = run_ps(random_index, display=True) # 랜덤으로 선택된 이미지만 표시 (display=True)
else:
    print("\n표시할 이미지가 없습니다. 'img' 폴더에 이미지가 있는지 확인해주세요.")
# --- 랜덤 이미지 표시 끝 ---

# --- 8. DPU 코드와 동일한 방식으로 성능 평가 수행 ---

# 모든 이미지에 대한 추론 및 성능 측정
print("\n--- 전체 이미지 Raspberry PI (CPU)기반 추론 성능 평가 ---")
time1 = time.time()
# 리스트 컴프리헨션으로 모든 이미지에 대해 추론 수행 (display=False로 속도 측정)
[run_ps(i, display=False) for i in range(total_images)] # 여기서는 속도 측정을 위해 display=False로 설정
time2 = time.time()

duration = time2 - time1
# total_images가 0이 아닐 경우에만 FPS 계산
fps_ps = 0.0
if duration > 0 and total_images > 0:
    fps_ps = total_images / duration

print("\nPS(CPU) 기반 성능 평가 및 랜덤 이미지 분류 결과 표시 완료.")
print("Performance PS(CPU): {} FPS".format(fps_ps))
print(f"총 걸린 시간: {duration:.2f}초") # 추가된 부분