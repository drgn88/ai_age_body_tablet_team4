import cv2
import tensorflow as tf
import numpy as np
import time, threading, subprocess

import builtins, functools
print = functools.partial(builtins.print, flush=True)  # 모든 print 즉시 출력


# 모델 경로
AGE_MODEL_PATH = "/home/linux/final/1_Age-VGG161.keras"
GENDER_MODEL_PATH = "/home/linux/final/1_Gender-ResNet1521.keras"

# ───── Ollama 환경 감지 ─────
USE_PY_API = False
try:
    import ollama 
    _ = ollama.list()          # 데몬 확인
    USE_PY_API = True
    print("[INFO] Ollama Python API 사용")
except Exception as e:
    print("[WARN] Python API 사용 불가, CLI fallback:", e)

OLLAMA_CLI    = "/usr/local/bin/ollama"  # CLI 위치
OLLAMA_MODEL  = "gemma3:1b"
OLLAMA_TIMEOUT = 120  # sec
# ───────────────────────────

# Ollama API 호출 함수 (이전과 동일)
def ask_ollama(prompt: str) -> str:
    """LLM 호출: Python API → CLI 순"""
    if USE_PY_API:
        try:
            res = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, stream=False)
            return res.get("response", "").strip()
        except Exception as e:
            return f"[API 오류] {e}"
    # CLI fallback
    try:
        p = subprocess.run([OLLAMA_CLI, "run", OLLAMA_MODEL, prompt],
                           text=True, capture_output=True, timeout=OLLAMA_TIMEOUT)
        return p.stdout.strip() if p.returncode == 0 else f"[CLI 오류] {p.stderr.strip()}"
    except Exception as e:
        return f"[CLI 예외] {e}"
    
# 모델 로드
try:
    age_model = tf.keras.models.load_model(AGE_MODEL_PATH, compile=False)
    gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH, compile=False)
    print("모델 로드 성공!")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    print("모델 파일 경로를 확인하거나, 모델 저장 시 사용된 custom_objects가 필요한지 확인해주세요.")
    exit()
                               
# ───────────────────────────
calling = False  # LLM 호출 중복 방지용 플래그
webcam_status_lock = threading.Lock() # 웹캠 상태 변경을 위한 락
webcam_needs_closure = False # 웹캠을 닫아야 함을 알리는 플래그
webcam_needs_reopen = False  # 웹캠을 다시 열어야 함을 알리는 플래그


# ollama_response_display 변수와 ollama_response_lock 제거 (화면 표시 안 함)

def worker(prompt):
    global calling
    global webcam_needs_closure
    global webcam_needs_reopen

    print("\n생각 중… 잠시만요")
    
    # 웹캠 닫기 신호를 보냄
    with webcam_status_lock:
        webcam_needs_closure = True

    # 웹캠이 닫힐 때까지 대기 (메인 루프에서 웹캠을 닫을 때까지 기다림)
    while True:
        with webcam_status_lock:
            if not webcam_needs_closure: # 웹캠 닫기 작업이 완료되면
                break
        time.sleep(0.01) # 짧게 대기

    reply = ask_ollama(prompt)       # ← Ollama 호출

    # 답변을 터미널에만 출력
    print("\n" + reply + "\n")

    calling = False
    
    # 웹캠 다시 열기 신호를 보냄
    with webcam_status_lock:
        webcam_needs_reopen = True

    
# 설정
IMAGE_SIZE = 224
gender_mapping = ["Male", "Female"]
# ───────────────────────────

# 얼굴 탐지기 로드 (OpenCV 기본 제공 HaarCascade 사용)
try: 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("경고: haarcascade_frontalface_default.xml 파일을 로드할 수 없습니다.")
        print(f"경로: {cv2.data.haarcascades + 'haarcascade_frontal_face_default.xml'} 확인.")
except Exception as e:
    print(f"Haar Cascade 로드 중 오류 발생: {e}")
    exit()

# 웹캠 초기 열기
cap = cv2.VideoCapture(0) # 0, 1, 2 등으로 카메라 인덱스를 바꿔가며 테스트해보세요.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("❌ 카메라 열기 실패. 카메라 인덱스를 확인하거나 다른 카메라를 시도해보세요.")

# FPS 고정을 위한 설정
padding, TARGET_FPS = 20, 20
FRAME_MS = int(1000 / TARGET_FPS)

# 예측 빈도 설정
PREDICTION_INTERVAL_FRAMES = 50 
frame_count = 0
last_predictions = {} # 각 얼굴 ID별 마지막 예측 결과를 저장

# Ollama 프롬프트용 변수
detected_age = None
detected_gender = None

print("스페이스바 → LLM 즉시 호출, q → 종료")

while True:
    start = time.time() # 프레임 처리 시작 시간 기록

    # 웹캠 닫기 요청 처리 (메인 스레드에서만 수행)
    with webcam_status_lock:
        if webcam_needs_closure:
            if cap and cap.isOpened():
                cap.release()
                print("웹캠 일시 정지.")
            webcam_needs_closure = False # 닫기 요청 처리 완료
            # 웹캠이 닫혔으니 이 프레임에서는 더 이상 이미지 처리를 하지 않음
            # 이 지점에서 검은 화면을 표시할 수도 있습니다.
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imshow("Age & Gender Prediction", black_frame)
            key = cv2.waitKey(delay) & 0xFF # 키 입력 대기 (창 업데이트 유지)
            continue # 다음 루프 반복으로 넘어가서 웹캠 재개 요청을 기다림

    # 웹캠 다시 열기 요청 처리 (메인 스레드에서만 수행)
    with webcam_status_lock:
        if webcam_needs_reopen:
            if not (cap and cap.isOpened()): # 웹캠이 실제로 닫혀있다면 다시 열기 시도
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                if not cap.isOpened():
                    print("❌ 웹캠 다시 열기 실패. 재시도합니다.")
                    # 실패 시에도 재시도 플래그를 해제하여 루프가 계속 돌게 함
                    webcam_needs_reopen = False 
                    # 웹캠이 열리지 않았으니 이 프레임에서는 이미지 처리를 하지 않음
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black_frame, "웹캠 열기 실패...", (100, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow("Age & Gender Prediction", black_frame)
                    key = cv2.waitKey(delay) & 0xFF # 키 입력 대기
                    continue
                else:
                    print("웹캠 재개.")
            webcam_needs_reopen = False # 재열기 요청 처리 완료
            # 웹캠이 열렸으니 다음 루프에서 바로 프레임을 읽도록 continue
            continue 

    # 웹캠이 열려있을 때만 프레임을 읽고 처리
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다. 웹캠 연결을 확인하세요. 프로그램 종료.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 감지 (매 프레임마다 수행)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5) 

        # 가장 큰 얼굴을 기준으로 나이와 성별을 설정 (간단화)
        main_face_area = 0
        main_face_id = -1

        for i, (x, y, w, h) in enumerate(faces):
            area = w * h
            if area > main_face_area:
                main_face_area = area
                main_face_id = i

            # 예측 빈도 로직 적용
            if frame_count % PREDICTION_INTERVAL_FRAMES == 0:
                # 설정된 간격마다 새로운 예측 수행
                face = frame[y:y+h, x:x+w]
                
                if face.shape[0] == 0 or face.shape[1] == 0:
                    continue

                image = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
                image = image / 255.0
                image = np.expand_dims(image, axis=0)

                try:
                    pred_age = age_model.predict(image, verbose=0)[0][0]
                    pred_gender = gender_model.predict(image, verbose=0)[0][0]
                    
                    age = int(pred_age)
                    gender = gender_mapping[int(np.round(pred_gender))]
                    
                    label = f"{gender}, Age: {age}"
                    last_predictions[i] = {'label': label, 'age': age, 'gender': gender} # 예측 결과 저장
                except Exception as e:
                    label = "Prediction Error"
                    print(f"예측 중 오류 발생 (얼굴 {i}): {e}")
                    last_predictions[i] = {'label': label, 'age': None, 'gender': None}
            else:
                # 예측 간격이 아니라면, 마지막으로 저장된 예측 결과 사용
                if i in last_predictions:
                    label = last_predictions[i]['label']
                else:
                    label = "Detecting..." # 새로 나타났거나 아직 예측되지 않은 얼굴

            # 얼굴 주위에 사각형 그리기
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 라벨 텍스트 추가
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 가장 큰 얼굴의 나이와 성별을 Ollama 프롬프트용으로 저장
        if main_face_id != -1 and main_face_id in last_predictions:
            detected_age = last_predictions[main_face_id].get('age')
            detected_gender = last_predictions[main_face_id].get('gender')
        else:
            detected_age = None
            detected_gender = None

        # 프레임 카운트 증가
        frame_count += 1
        # ──────────────────────────────────────────

        cv2.imshow("Age & Gender Prediction", frame)
    else:
        # 웹캠이 닫혀있을 때 (혹은 아직 열리지 않았을 때) 검은 화면 표시
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 웹캠이 닫혔다는 메시지 표시 (이미 'Ollama 답변 준비 중...'이 표시될 수 있음)
        # 이 부분은 필요에 따라 조정 가능
        cv2.imshow("Age & Gender Prediction", black_frame)
    
    delay = max(1, FRAME_MS - int((time.time()-start)*1000))
    key = cv2.waitKey(delay) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        if calling:
            print("[INFO] 이미 추천을 생성 중입니다. 잠시만 기다려 주세요!")
            continue                         # 새 스레드 만들지 않음
        if detected_gender and detected_age:
            calling = True
            llm_reply = "🔄 영양제 추천 생성 중…"
            prompt = (f"나는 {detected_age} 살 {detected_gender} 성별의 한국인이야."
                      f"필요한 주요 영양제 3가지를 알려주고,"
                      f"각 영양제를 권장하는 이유를 5줄로 설명해 줘. 답변은 한국어로만 해줘.") # 한국어 답변 요청 추가
            print("[LLM 요청] 생성 요청:", prompt) # CLI에 요청 프롬프트도 출력
            
            # worker 스레드 시작 (데몬 스레드로 설정하여 메인 프로그램 종료 시 함께 종료되도록)
            threading.Thread(target=worker, args=(prompt,), daemon=True).start()
        else:
            print("[WARN] 나이 또는 성별 정보가 감지되지 않아 Ollama를 호출할 수 없습니다.")


# 자원 해제
if cap and cap.isOpened(): # 프로그램 종료 시 웹캠이 열려있으면 닫기
    cap.release()
cv2.destroyAllWindows()

print("프로그램 종료.")