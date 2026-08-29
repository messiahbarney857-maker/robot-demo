# demo_robot.py
# Upgraded demo: Vision (YOLOv8) + Speech (mic -> OpenAI Chat) + TTS + Behavior/Mood
# Set OPENAI_API_KEY env var (or create a .env file) to enable ChatGPT-style replies.

import os
import time
import threading
import queue
import cv2
import numpy as np
import pyttsx3
import openai
from dotenv import load_dotenv

from ultralytics import YOLO  # pip install ultralytics

# Optional local speech-to-text (Whisper). If not installed, fallback to typed input.
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# Load .env if present
load_dotenv()

# ---------- Configuration ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # required for ChatGPT-style replies
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

YOLO_MODEL = "yolov8n.pt"
CAMERA_INDEX = 0
SPEECH_TIMEOUT = 5
ANNOUNCE_COOLDOWN = 3.0

# ---------- Simple MoodManager ----------
class MoodManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.moods = {"curiosity": 0.5, "friendliness": 0.7, "fatigue": 0.1}
        self.last_update = time.time()

    def update_from_event(self, event):
        with self.lock:
            t = event.get("type")
            if t == "see_novel_object":
                self.moods["curiosity"] = min(1.0, self.moods["curiosity"] + 0.15)
                self.moods["fatigue"] = min(1.0, self.moods["fatigue"] + 0.02)
            elif t == "positive_interaction":
                self.moods["friendliness"] = min(1.0, self.moods["friendliness"] + 0.1)
                self.moods["fatigue"] = max(0.0, self.moods["fatigue"] - 0.05)
            elif t == "time_pass":
                self.moods["curiosity"] = max(0.0, self.moods["curiosity"] - 0.01)
                self.moods["fatigue"] = min(1.0, self.moods["fatigue"] + 0.005)
            self.last_update = time.time()

    def snapshot(self):
        with self.lock:
            return dict(self.moods)

# ---------- Chat session wrapper (conversation history) ----------
class ChatSession:
    def __init__(self, system_prompt="You are a helpful robot assistant. Keep replies short and action-oriented."):
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    def ask(self, user_text, max_tokens=200, temperature=0.8):
        if not OPENAI_API_KEY:
            return None
        self.messages.append({"role": "user", "content": user_text})
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            assistant_text = resp.choices[0].message["content"].strip()
            self.messages.append({"role": "assistant", "content": assistant_text})
            return assistant_text
        except Exception as e:
            print("OpenAI error:", e)
            return "Sorry, I couldn't reach the language model."

# ---------- Behavior Engine (delegates to LLM when needed) ----------
class BehaviorEngine:
    def __init__(self, mood_manager, speaker, chat_session=None):
        self.mood_manager = mood_manager
        self.speaker = speaker
        self.chat_session = chat_session
        self.last_announced = {}

    def on_detection(self, detections):
        now = time.time()
        mood = self.mood_manager.snapshot()
        actions = []
        for name, conf, bbox in detections:
            last = self.last_announced.get(name, 0)
            if now - last > ANNOUNCE_COOLDOWN:
                if mood["curiosity"] > 0.6:
                    actions.append(("announce", f"I see a {name}. I'm curious about it!"))
                elif mood["friendliness"] > 0.6:
                    actions.append(("announce", f"Hey! That's a {name}. Nice."))
                else:
                    actions.append(("announce", f"{name} detected."))
                self.last_announced[name] = now
                self.mood_manager.update_from_event({"type":"see_novel_object"})
        return actions

    def on_command(self, text):
        mood = self.mood_manager.snapshot()
        text_lower = text.lower()

        # quick rule-based replies
        if "hello" in text_lower or "hi" in text_lower:
            self.mood_manager.update_from_event({"type":"positive_interaction"})
            return [("speak", f"Hello! I'm feeling {self._mood_word(mood)}. How can I help?")]
        if "what do you see" in text_lower or "what is there" in text_lower:
            return [("speak", "I can look and tell you what I see. Ask me to scan the scene.")]
        if "sleep" in text_lower or "rest" in text_lower:
            self.mood_manager.update_from_event({"type":"time_pass"})
            return [("speak", "Okay, I'll rest for a bit.")]

        # delegate to LLM if available
        if OPENAI_API_KEY and self.chat_session:
            return [("llm", text)]
        # fallback to speak that we can't reach LLM
        return [("speak", "I don't have access to the chat service. I can still respond with simple behaviors.")]

    def _mood_word(self, mood):
        if mood["fatigue"] > 0.7:
            return "tired"
        if mood["curiosity"] > 0.7:
            return "curious"
        if mood["friendliness"] > 0.7:
            return "friendly"
        return "okay"

# ---------- Speaker (TTS) ----------
class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    def speak(self, text):
        print("[SPEAK]", text)
        # start say and runAndWait in background to avoid blocking main loop
        t = threading.Thread(target=self._speak_blocking, args=(text,))
        t.daemon = True
        t.start()

    def _speak_blocking(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

# ---------- Perception (YOLO) ----------
class Perception:
    def __init__(self, model_name=YOLO_MODEL):
        print("Loading YOLO model (may download weights)...")
        self.model = YOLO(model_name)

    def detect(self, frame):
        results = self.model.predict(frame, imgsz=640, conf=0.35, max_det=5)
        detections = []
        if len(results):
            r = results[0]
            boxes = getattr(r, "boxes", [])
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = self.model.model.names.get(cls_id, str(cls_id))
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                detections.append((name, conf, xyxy))
        return detections

# ---------- STT helpers ----------
def transcribe_with_whisper(audio_path):
    if not WHISPER_AVAILABLE:
        return None
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    return result.get("text", "").strip()

def record_audio_to_file(filename, duration=SPEECH_TIMEOUT, samplerate=16000):
    import sounddevice as sd
    from scipy.io.wavfile import write
    print("Recording audio for", duration, "s...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    write(filename, samplerate, audio)
    print("Saved audio to", filename)

# ---------- Main application ----------
def main_loop():
    mood = MoodManager()
    speaker = Speaker()
    chat_session = ChatSession() if OPENAI_API_KEY else None
    behavior = BehaviorEngine(mood, speaker, chat_session)
    perception = Perception()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    stt_queue = queue.Queue()

    def listen_loop():
        while True:
            audio_file = "temp_cmd.wav"
            record_audio_to_file(audio_file, duration=SPEECH_TIMEOUT)
            text = None
            if WHISPER_AVAILABLE:
                text = transcribe_with_whisper(audio_file)
            else:
                print("Whisper not available; type the command (or leave blank):")
                text = input(">> ").strip()
            if text:
                print("[USER SAID]", text)
                stt_queue.put(text)
            time.sleep(0.1)

    threading.Thread(target=listen_loop, daemon=True).start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = perception.detect(frame)
            for name, conf, xyxy in detections:
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            if detections:
                actions = behavior.on_detection(detections)
                for act, payload in actions:
                    if act == "announce":
                        speaker.speak(payload)

            while not stt_queue.empty():
                cmd_text = stt_queue.get_nowait()
                acts = behavior.on_command(cmd_text)
                for act, payload in acts:
                    if act == "speak":
                        speaker.speak(payload)
                    elif act == "llm":
                        # query chat session
                        reply = None
                        if chat_session:
                            prompt = f"The user said: \"{payload}\". Keep reply short and friendly."
                            reply = chat_session.ask(prompt)
                        if reply:
                            speaker.speak(reply)
                        else:
                            speaker.speak("I couldn't contact the chat service; try again later.")

            m = mood.snapshot()
            cv2.putText(frame, f"Mood: cur={m['curiosity']:.2f} fr={m['friendliness']:.2f} ft={m['fatigue']:.2f}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.imshow("Robot Demo (ChatGPT-enabled)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            mood.update_from_event({"type":"time_pass"})
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
