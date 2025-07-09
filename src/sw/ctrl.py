#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama 대화 + 웹캠 컨트롤러

• 터미널에서 Ollama에게 자연어로 묻는다
• 사용자가 Ollama에게 “웹캠실행”이라 말하면 import_cv2.py 실행
• “웹캠종료”라 말하면 종료
• import_cv2.py 안 : 스페이스바 → 영양제 추천,  q → 창 닫기
• 'q'·'종료' 입력 시 이 컨트롤러 전체 종료
"""

import subprocess, threading, shlex, re, sys
import ollama        # pip install ollama
from pathlib import Path

# ───── 사용자 환경에 맞게 수정 ─────
MODEL      = "gemma3:1b"                         # Ollama 모델명
WEBCAM_CMD = "python /home/linux/final/import_cv2.py"  # 가상환경·절대경로 필요 시 수정
# ────────────────────────────────

webcam_proc   = None           # 웹캠 서브프로세스 핸들
conversation  = []             # Ollama 대화 기록

# ───── 웹캠 프로세스 ─────
def _relay(pipe, tag):
    for line in iter(pipe.readline, ''):
        print(f"[{tag}] {line.rstrip()}")
    pipe.close()

def start_webcam():
    global webcam_proc
    if webcam_proc and webcam_proc.poll() is None:
        print("✅ 이미 웹캠이 켜져 있습니다.")
        return
    print("▶ import_cv2.py 실행…")
    webcam_proc = subprocess.Popen(
        shlex.split(WEBCAM_CMD),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    threading.Thread(target=_relay, args=(webcam_proc.stdout, "CAM"), daemon=True).start()
    threading.Thread(target=_relay, args=(webcam_proc.stderr, "CAM_ERR"), daemon=True).start()

def stop_webcam():
    global webcam_proc
    if webcam_proc and webcam_proc.poll() is None:
        print("■ 웹캠 종료 중…")
        webcam_proc.terminate()
        try:
            webcam_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            webcam_proc.kill()
        print("🛑 웹캠 종료 완료.")
    else:
        print("ℹ️ 실행 중인 웹캠이 없습니다.")

# ───── Ollama 래퍼 ─────
def ask_ollama(user_msg: str) -> str:
    conversation.append({"role": "user", "content": user_msg})
    try:
        res = ollama.chat(model=MODEL, messages=conversation, stream=False, keep_alive=120)
        reply = res["message"]["content"].strip()
    except Exception as e:
        reply = f"[Ollama 오류] {e}"
    conversation.append({"role": "assistant", "content": reply})
    return reply

CMD_ON  = re.compile(r"웹캠\s*실행", re.I)
CMD_OFF = re.compile(r"웹캠\s*종료", re.I)

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("Ollama 컨트롤러 시작!")
print("  Ollama에게 물어보다가 ‘웹캠실행’ 하면 카메라 ON")
print("  ‘웹캠종료’ 하면 카메라 OFF")
print("  ‘q’ 또는 ‘종료’ 입력 시 프로그램 종료")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

try:
    while True:
        user = input("나> ").strip()
        if user.lower() in {"q", "quit", "종료", "exit"}: 
            break

        reply = ask_ollama(user)
        print(f"\n[Ollama]\n{reply}\n")

        # 사용자 발화에 명령어 포함?
        if CMD_ON.search(user):
            start_webcam()
        elif CMD_OFF.search(user):
            stop_webcam()

        # Ollama 응답에 명령어 포함?
        if CMD_ON.search(reply):
            start_webcam()
        elif CMD_OFF.search(reply):
            stop_webcam()

except (KeyboardInterrupt, EOFError):
    print()

# 프로그램 종료 시 정리
stop_webcam()
print("컨트롤러 종료.")
