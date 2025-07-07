#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import time
import numpy as np
import os

# --- 1. YOLO 모델 파일 경로 및 클래스 이름 설정 ---
# 당신의 Ultra96v2 보드에 이 파일들을 업로드하고 경로를 맞게 설정해주세요.
# ONNX 대신 .cfg와 .weights 파일을 사용합니다.
model_cfg_path = "yolov3.cfg"
model_weights_path = "yolov3.weights"
class_names_path = "coco.names" # COCO 데이터셋 기반이라면

# 모델 파일이 존재하지 않으면 오류 메시지 출력 및 종료
if not os.path.exists(model_cfg_path):
    print(f"오류: YOLO CFG 모델 파일 '{model_cfg_path}'을(를) 찾을 수 없습니다.")
    print("Ultra96v2 보드에 모델 파일을 업로드하고 경로를 확인해주세요.")
    raise SystemExit("CFG 파일 누락.")

if not os.path.exists(model_weights_path):
    print(f"오류: YOLO Weights 모델 파일 '{model_weights_path}'을(를) 찾을 수 없습니다.")
    print("Ultra96v2 보드에 모델 파일을 업로드하고 경로를 확인해주세요.")
    raise SystemExit("Weights 파일 누락.")

if not os.path.exists(class_names_path):
    print(f"오류: 클래스 이름 파일 '{class_names_path}'을(를) 찾을 수 없습니다.")
    print("Ultra96v2 보드에 클래스 이름 파일을 업로드하고 경로를 확인해주세요.")
    raise SystemExit("클래스 이름 파일 누락.")

# --- 2. YOLO 모델 로드 (OpenCV DNN 모듈 사용) ---
print(f"YOLO 모델을 OpenCV DNN으로 로드 중: {model_cfg_path}, {model_weights_path}...")
try:
    # readNetFromONNX 대신 readNetFromDarknet 사용
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)
    
    # PS (CPU)에서 추론하도록 설정합니다.
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("YOLO 모델이 성공적으로 로드되었으며 CPU 추론으로 설정되었습니다.")
except Exception as e:
    print(f"오류: YOLO 모델 로드 실패: {e}")
    print("CFG/Weights 파일이 올바르고 OpenCV DNN 모듈에서 지원되는 형식인지 확인해주세요.")
    raise SystemExit("모델 로드 실패.")

# --- 3. 클래스 이름 로드 ---
class_names = []
with open(class_names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# --- 4. YOLO 관련 설정 ---
# YOLO 모델의 입력 크기 (모델에 따라 416x416, 608x608 등으로 변경될 수 있습니다.)
model_input_size = (416, 416) 

# YOLO 모델의 출력 레이어 이름을 가져옵니다.
output_layers_names = net.getUnconnectedOutLayersNames()

# --- 5. 평가 관련 유틸리티 함수 정의 ---

def pre_process_for_ps(image, input_size):
    """
    OpenCV DNN에 적합한 이미지 전처리 (리사이징, 스케일링 등).
    Darknet YOLO는 보통 BGR 입력을 받으므로 swapRB=False를 유지하거나,
    모델에 따라 swapRB=True로 BGR->RGB 변환이 필요할 수 있습니다.
    대부분의 Pretrained Darknet 모델은 BGR을 기대합니다.
    """
    # Darknet 모델은 보통 BGR 입력을 기대하며, 1/255.0 스케일링만 합니다.
    blob = cv2.dnn.blobFromImage(image, 1/255.0, input_size, swapRB=False, crop=False)
    return blob

def post_process_yolo_output(net_outputs, image_shape, score_threshold=0.5, nms_threshold=0.4):
    """
    OpenCV DNN YOLO 출력에서 바운딩 박스, 점수, 클래스를 추출하고 NMS를 적용합니다.
    """
    height, width = image_shape
    boxes, confidences, class_ids = [], [], []

    for output in net_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > score_threshold:
                # YOLOv3/v4 출력은 중심 좌표, 너비, 높이
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 상단 왼쪽 좌표로 변환
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 비최대 억제 (Non-Maximum Suppression) 적용
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    
    if len(indices) > 0:
        indices = indices.flatten()
        final_boxes = np.array(boxes)[indices]
        final_scores = np.array(confidences)[indices]
        final_classes = np.array(class_ids)[indices]
    else:
        final_boxes = np.array([])
        final_scores = np.array([])
        final_classes = np.array([])

    return final_boxes, final_scores, final_classes

def draw_bboxes_on_image(image, bboxes_scores_classes, class_names, colors=None):
    """
    이미지에 바운딩 박스, 점수, 클래스 레이블을 그립니다.
    DPU 코드의 draw_boxes 함수와 호환되도록 bboxes_scores_classes를 통합하여 받습니다.
    bboxes_scores_classes는 [x, y, w, h, score, class_id] 형태로 구성된 2D numpy 배열이어야 합니다.
    """
    if colors is None:
        np.random.seed(42)
        colors = np.random.uniform(0, 255, size=(len(class_names), 3)).astype(int)

    result_image = image.copy()
    
    if bboxes_scores_classes.size == 0:
        return result_image

    for detection in bboxes_scores_classes:
        x, y, w, h = map(int, detection[:4])
        score = detection[4]
        class_id = int(detection[5])

        label = str(class_names[class_id])
        color = (int(colors[class_id, 0]), int(colors[class_id, 1]), int(colors[class_id, 2]))

        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(result_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return result_image

# --- 6. 테스트 이미지 경로 설정 (DPU 코드와 동일하게) ---
image_folder = 'img' # 당신의 이미지 폴더 경로
if not os.path.exists(image_folder):
    print(f"오류: 이미지 폴더 '{image_folder}'을(를) 찾을 수 없습니다.")
    print("DPU 코드와 동일하게 Ultra96v2 보드에 'test_images' 폴더를 만들어주세요.")
    raise SystemExit("이미지 폴더 누락.")

original_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
original_images.sort() # 순서 보장
total_images = len(original_images)

if total_images == 0:
    print(f"오류: '{image_folder}' 폴더에 이미지 파일이 없습니다. 이미지를 추가해주세요.")
    raise SystemExit("테스트 이미지 없음.")

print(f"'{image_folder}'에서 {total_images}개의 테스트 이미지를 찾았습니다.")


# --- 7. PS (CPU) 추론 실행 함수 (DPU 코드의 run 함수와 유사하게 재구성) ---
def run_ps(image_index, display=False):
    """
    지정된 이미지에 대해 PS (CPU) 기반 YOLOv3 추론을 수행합니다.
    DPU 코드의 run 함수와 동일한 인터페이스를 가집니다.
    """
    # 1. 이미지 로드
    image_path = os.path.join(image_folder, original_images[image_index])
    input_image = cv2.imread(image_path)
    if input_image is None:
        print(f"경고: 이미지 '{image_path}'을(를) 읽을 수 없습니다. 건너뜀.")
        print("Number of detected objects: 0")
        return 0, np.array([]) 

    image_size = input_image.shape[:2] 

    # 2. 이미지 전처리
    blob = pre_process_for_ps(input_image, model_input_size)
    net.setInput(blob)

    # 3. 추론 실행
    net_outputs = net.forward(output_layers_names)
    
    # 4. 출력 후처리
    boxes, scores, classes = post_process_yolo_output(net_outputs, image_size)
    
    # DPU 코드의 draw_boxes 함수와 호환되는 형식으로 데이터를 통합 (x,y,w,h,score,class_id)
    if len(boxes) > 0:
        bboxes_scores_classes = np.concatenate((boxes, scores[:, np.newaxis], classes[:, np.newaxis]), axis=1)
    else:
        bboxes_scores_classes = np.array([]) 

    if display:
        from IPython.display import display, Image # 함수 내부에 import
        _ = draw_bboxes_on_image(input_image.copy(), bboxes_scores_classes, class_names)
        if _ is not None:
            max_display_width = 800
            if _.shape[1] > max_display_width:
                scale_factor = max_display_width / _.shape[1]
                _ = cv2.resize(_, (max_display_width, int(_.shape[0] * scale_factor)))
            _, buffer = cv2.imencode('.jpg', _)
            display(Image(data=buffer.tobytes()))
            
    print("Number of detected objects: {}".format(len(boxes)))
    return len(boxes), bboxes_scores_classes 


# --- 8. DPU 코드와 동일한 방식으로 성능 평가 수행 ---

# 단일 이미지 테스트 및 표시 (DPU 코드의 In[21] 셀과 동일)
print("\n--- 단일 이미지 PS (CPU) 추론 테스트 ---")
# 예시: 인덱스 2의 이미지
_ , _ = run_ps(2, display=True) 


# 모든 이미지에 대한 추론 및 성능 측정 (DPU 코드의 In[22] 셀과 동일)
print("\n--- 전체 이미지 PS (CPU) 추론 성능 평가 ---")
time1 = time.time()
# 리스트 컴프리헨션으로 모든 이미지에 대해 추론 수행 (display=False로 속도 측정)
[run_ps(i) for i in range(total_images)]
time2 = time.time()

duration = time2 - time1
# total_images가 0이 아닐 경우에만 FPS 계산
fps_ps = 0.0
if duration > 0 and total_images > 0:
    fps_ps = total_images / duration

print("Performance (PS/CPU): {} FPS".format(fps_ps))

print("\nPS (CPU) 기반 성능 평가 완료.")


# In[ ]:




