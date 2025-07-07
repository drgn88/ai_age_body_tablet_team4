#!/usr/bin/env python
# coding: utf-8

# # DPU example: YOLOv3
# ----

# ## Aim/s
# * This notebooks shows an example of DPU applications. The application,as well as the DPU IP, is pulled from the official 
# [Vitis AI Github Repository](https://github.com/Xilinx/Vitis-AI).
# 
# ## References
# * [Vitis AI Github Repository](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html).
# 
# ## Last revised
# * Jun 27, 2022
#     * Initial revision
# ----

# ## 1. Prepare the overlay
# We will download the overlay onto the board. 

# In[1]:


from pynq_dpu import DpuOverlay
overlay = DpuOverlay("dpu.bit")


# ## 2. Utility functions
# 
# In this section, we will prepare a few functions for later use.

# In[2]:


import os
import time
import numpy as np
import cv2
import random
import colorsys
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# The `load_model()` method will automatically prepare the `graph`
# which is used by VART.
# 
# **Note** For the KV260 board you may see TLS memory allocation errors if cv2 gets loaded before loading the vitis libraries in the Jupyter Lab environment. Make sure to load cv2 first in these cases.

# In[3]:


overlay.load_model("tf_yolov3_voc.xmodel")


# Let's first define a few useful preprocessing functions.

# In[4]:


anchor_list = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
anchor_float = [float(x) for x in anchor_list]
anchors = np.array(anchor_float).reshape(-1, 2)


# In[5]:


'''Get model classification information'''	
def get_class(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
    
classes_path = "img/voc_classes.txt"
class_names = get_class(classes_path)


# In[6]:


num_classes = len(class_names)
hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: 
                  (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), 
                  colors))
random.seed(0)
random.shuffle(colors)
random.seed(None)


# In[7]:


'''resize image with unchanged aspect ratio using padding'''
def letterbox_image(image, size):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    #print(scale)
    
    nw = int(iw*scale)
    nh = int(ih*scale)
    #print(nw)
    #print(nh)

    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h,w,3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image


'''image preprocessing'''
def pre_process(image, model_image_size):
    image = image[...,::-1]
    image_h, image_w, _ = image.shape
 
    if model_image_size != (None, None):
        assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) 	
    return image_data


# We will also define a few functions to post-process the output after running a DPU task.

# In[8]:


def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    nu = num_classes + 5
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis = -1)
    grid = np.array(grid, dtype=np.float32)

    box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
    box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
    box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
    return box_xy, box_wh, box_confidence, box_class_probs


def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, dtype = np.float32)
    image_shape = np.array(image_shape, dtype = np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxes[..., 0:1],
        box_maxes[..., 1:2]
    ], axis = -1)
    boxes *= np.concatenate([image_shape, image_shape], axis = -1)
    return boxes


def boxes_and_scores(feats, anchors, classes_num, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = _get_feats(feats, anchors, classes_num, input_shape)
    boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, [-1, classes_num])
    return boxes, box_scores


# In[9]:


'''Draw detection frame'''
def draw_bbox(image, bboxes, classes):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
    return image


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.55)[0]  # threshold
        order = order[inds + 1]

    return keep


# In[10]:


def draw_boxes(image, boxes, scores, classes):
    _, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_h, image_w, _ = image.shape

    for i, bbox in enumerate(boxes):
        [top, left, bottom, right] = bbox
        width, height = right - left, bottom - top
        center_x, center_y = left + width*0.5, top + height*0.5
        score, class_index = scores[i], classes[i]
        label = '{}: {:.4f}'.format(class_names[class_index], score) 
        color = tuple([color/255 for color in colors[class_index]])
        ax.add_patch(Rectangle((left, top), width, height,
                               edgecolor=color, facecolor='none'))
        ax.annotate(label, (center_x, center_y), color=color, weight='bold', 
                    fontsize=12, ha='center', va='center')
    return ax


# In[11]:


def evaluate(yolo_outputs, image_shape, class_names, anchors):
    score_thresh = 0.2
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    boxes = []
    box_scores = []
    input_shape = np.shape(yolo_outputs[0])[1 : 3]
    input_shape = np.array(input_shape)*32

    for i in range(len(yolo_outputs)):
        _boxes, _box_scores = boxes_and_scores(
            yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), 
            input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis = 0)
    box_scores = np.concatenate(box_scores, axis = 0)

    mask = box_scores >= score_thresh
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(len(class_names)):
        class_boxes_np = boxes[mask[:, c]]
        class_box_scores_np = box_scores[:, c]
        class_box_scores_np = class_box_scores_np[mask[:, c]]
        nms_index_np = nms_boxes(class_boxes_np, class_box_scores_np) 
        class_boxes_np = class_boxes_np[nms_index_np]
        class_box_scores_np = class_box_scores_np[nms_index_np]
        classes_np = np.ones_like(class_box_scores_np, dtype = np.int32) * c
        boxes_.append(class_boxes_np)
        scores_.append(class_box_scores_np)
        classes_.append(classes_np)
    boxes_ = np.concatenate(boxes_, axis = 0)
    scores_ = np.concatenate(scores_, axis = 0)
    classes_ = np.concatenate(classes_, axis = 0)

    return boxes_, scores_, classes_

# ... (기존 import 및 초기 설정 코드) ...

import cv2 # OpenCV 임포트 추가
from IPython.display import display, Image, clear_output

# DPU 모델의 입력 이미지 크기를 정의합니다.
model_input_size = (416, 416) # 실제 모델의 입력 크기로 변경하세요

# 웹캠 초기화 (cv2.VideoCapture 사용)
# 0은 일반적으로 기본 웹캠을 의미합니다. 다른 USB 웹캠이라면 숫자를 변경하세요.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Please ensure your USB webcam is connected and recognized by the system.")
    # 시스템에서 USB 웹캠이 인식되는지 확인하는 명령: ls /dev/video*
    exit()

print("Webcam streaming started...")
print("Press Ctrl+C to quit the webcam feed.")


# VART DPU Runner 관련 초기화 (이전 코드에서 그대로 가져옴)
dpu = overlay.runner
inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()
shapeIn = tuple(inputTensors[0].dims)
shapeOut0 = (tuple(outputTensors[0].dims))
shapeOut1 = (tuple(outputTensors[1].dims))
shapeOut2 = (tuple(outputTensors[2].dims))

input_data_buffer = [np.empty(shapeIn, dtype=np.float32, order="C")]
output_data_buffers = [
    np.empty(shapeOut0, dtype=np.float32, order="C"),
    np.empty(shapeOut1, dtype=np.float32, order="C"),
    np.empty(shapeOut2, dtype=np.float32, order="C")
]

# 비동기 처리를 위한 변수
current_job_id = None
previous_frame_for_display = None

try:
    while True:
        # 1. 이전 DPU 작업이 있었다면 결과 가져오기 및 처리
        if current_job_id is not None:
            dpu.wait(current_job_id)

            # 이전 DPU 작업 결과 (output_data_buffers에 채워짐) 가져와서 후처리
            conv_out0 = np.reshape(output_data_buffers[0], shapeOut0)
            conv_out1 = np.reshape(output_data_buffers[1], shapeOut1)
            conv_out2 = np.reshape(output_data_buffers[2], shapeOut2)
            yolo_outputs = [conv_out0, conv_out1, conv_out2]

            if previous_frame_for_display is not None:
                boxes, scores, classes = evaluate(yolo_outputs, previous_frame_for_display.shape[:2], class_names, anchors)
                
                # DPU 추론 결과를 원본 프레임에 그리기
                # draw_bbox 함수는 OpenCV 기반이므로 cv2.imshow 대용으로 사용 가능
                result_frame = draw_bbox(previous_frame_for_display.copy(), # 원본 프레임을 복사하여 수정
                                         np.concatenate((boxes, scores[:, np.newaxis], classes[:, np.newaxis]), axis=1), 
                                         class_names)

                # Jupyter Lab에 결과 프레임 표시
                _, buffer = cv2.imencode('.jpg', result_frame)
                i = Image(data=buffer.tobytes())
                display(i)
                clear_output(wait=True) # 이전 출력 지우기
            
            current_job_id = None # Job 완료했으므로 초기화

        # 2. 웹캠에서 새 프레임 읽기 (이 작업은 CPU에서 수행)
        ret, frame = cap.read() # cv2.VideoCapture의 read()는 (ret, frame)을 반환
        if not ret:
            print("Failed to read frame from webcam. Exiting.")
            break

        # 3. 새 프레임 전처리 (이 작업은 CPU에서 수행)
        previous_frame_for_display = frame.copy() # 원본 프레임을 다음 반복에서 DPU 결과 표시용으로 저장
        
        # pre_process 함수는 BGR을 RGB로 변환하는 로직을 포함해야 합니다.
        # 기존 pre_process 함수에서 image = image[...,::-1] 부분이 이 역할을 합니다.
        processed_image = pre_process(frame, model_input_size)
        input_data_buffer[0][...] = processed_image.reshape(shapeIn[1:])

        # 4. DPU에 새 작업 비동기적으로 제출
        current_job_id = dpu.execute_async(input_data_buffer, output_data_buffers)

        # 짧은 대기 (선택 사항): CPU가 너무 빨리 다음 프레임을 처리하는 것을 방지
        # time.sleep(0.01)

except KeyboardInterrupt:
    print("Webcam streaming stopped by user.")
finally:
    # 웹캠 해제
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    
    # DPU 및 Overlay 해제 (PYNQ Jupyter 노트북 환경에서 Kernel Restart 시 자동으로 해제되지만 명시적 해제 권장)
    if 'dpu' in locals():
        del dpu
    if 'overlay' in locals():
        del overlay
    
    print("Webcam and DPU resources released.")
