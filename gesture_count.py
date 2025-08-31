#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thumbs-Up ("いいね") counter with:
- MediaPipe Tasks Gesture Recognizer (VIDEO mode, num_hands=1)
- Confetti + red heart + floating '+1' effects
- Random WAV playback from an audio folder (winsound; no extra deps)
- Lightweight processing (640x480, ~15 FPS)
- 3-second cooldown after each like

Usage:
  python gesture_count.py --model assets/gesture_recognizer.task
Options:
  --camera 0            Camera index
  --width 640 --height 480
  --audio-dir audio     Folder containing WAVs (randomly played)
  --no-sound            Disable sound
  --show-fps            Show instantaneous FPS
Keys:
  q=quit, r=reset counter
"""
import argparse
import time
import sys
import os
import math
import glob
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python  # noqa: F401
from mediapipe.tasks.python import vision

# ------------------------- Config -------------------------
PROCESS_FPS = 15.0                 # Target processing FPS (skip frames to match)
DRAW_LANDMARKS = False
LIKE_LABEL = "Thumb_Up"
SCORE_THRESHOLD = 0.6
COOLDOWN_SEC = 2.0                 # ★ After a like, ignore next detections for 3 seconds
EFFECT_LIFETIME = 0.6
EFFECT_PARTICLES = 40

# ------------------------- Effects -------------------------
@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    t_end: float

@dataclass
class EffectState:
    particles: List[Particle] = field(default_factory=list)
    hearts: List[Tuple[int, int, float]] = field(default_factory=list)  # (cx, cy, t_end)
    texts: List[Tuple[int, int, float]] = field(default_factory=list)   # (cx, cy, t_end)

    def trigger(self, cx: float, cy: float, now: float, w: int, h: int):
        self.particles.clear()
        self.hearts.clear()
        self.texts.clear()
        # confetti burst
        for i in range(EFFECT_PARTICLES):
            angle = 2 * math.pi * (i / EFFECT_PARTICLES) + np.random.uniform(-0.2, 0.2)
            speed = np.random.uniform(120, 240)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append(Particle(cx, cy, vx, vy, now + EFFECT_LIFETIME))
        # heart (center) and '+1' slightly above
        self.hearts.append((int(cx), int(cy), now + EFFECT_LIFETIME))
        self.texts.append((int(cx), int(cy - 40), now + EFFECT_LIFETIME))

    def draw(self, frame: np.ndarray, now: float):
        if not (self.particles or self.hearts or self.texts):
            return
        h, w = frame.shape[:2]

        # particles
        alive_particles = []
        for p in self.particles:
            if now <= p.t_end:
                prog = 1.0 - (p.t_end - now) / EFFECT_LIFETIME
                x = int(p.x + p.vx * prog * EFFECT_LIFETIME)
                y = int(p.y + p.vy * prog * EFFECT_LIFETIME)
                if 0 <= x < w and 0 <= y < h:
                    radius = max(1, int(6 * (1.0 - prog) + 1))
                    cv2.circle(frame, (x, y), radius, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                alive_particles.append(p)
        self.particles = alive_particles

        # heart (parametric heart curve, filled red)
        alive_hearts = []
        for (cx, cy, t_end) in self.hearts:
            if now <= t_end:
                prog = 1.0 - (t_end - now) / EFFECT_LIFETIME
                scale = 1.0 + 0.5 * prog  # grow a bit over time
                pts = []
                for theta in np.linspace(0, 2 * math.pi, 100):
                    x = 16 * (math.sin(theta) ** 3)
                    y = -(13 * math.cos(theta) - 5 * math.cos(2 * theta) - 2 * math.cos(3 * theta) - math.cos(4 * theta))
                    pts.append((int(cx + x * scale), int(cy + y * scale)))
                pts = np.array(pts, np.int32)
                cv2.fillPoly(frame, [pts], (0, 0, 255))
                alive_hearts.append((cx, cy, t_end))
        self.hearts = alive_hearts

        # "+1" floating up with drop-shadow
        alive_texts = []
        for (tx, ty, t_end) in self.texts:
            if now <= t_end:
                prog = 1.0 - (t_end - now) / EFFECT_LIFETIME
                y = int(ty - 40 * prog)
                cv2.putText(frame, "+1", (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, "+1", (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
                alive_texts.append((tx, ty, t_end))
        self.texts = alive_texts

# ------------------------- Sound -------------------------
def play_random_wav(folder: str):
    """Play a random WAV from the given folder using winsound (Windows). Silently ignore on failure/empty."""
    if not folder or not os.path.isdir(folder):
        return
    files = glob.glob(os.path.join(folder, "*.wav"))
    if not files:
        return
    path = random.choice(files)
    try:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass  # keep silent on errors

# ------------------------- Utils -------------------------
def norm_rect_from_landmarks(landmarks, w: int, h: int) -> Tuple[int, int, int, int]:
    xs = [lm.x for lm in landmarks]; ys = [lm.y for lm in landmarks]
    if not xs or not ys:
        return (0, 0, w, h)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = 0.05
    x0 = max(0.0, x0 - pad); y0 = max(0.0, y0 - pad)
    x1 = min(1.0, x1 + pad); y1 = min(1.0, y1 + pad)
    return int(x0 * w), int(y0 * h), int((x1 - x0) * w), int((y1 - y0) * h)

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="assets/gesture_recognizer.task", help="Path to gesture_recognizer.task")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--width", type=int, default=640, help="Capture width")
    ap.add_argument("--height", type=int, default=480, help="Capture height")
    ap.add_argument("--audio-dir", default="audio", help="Folder of WAVs to play randomly on like")
    ap.add_argument("--no-sound", action="store_true", help="Disable sound")
    ap.add_argument("--show-fps", action="store_true", help="Overlay instantaneous FPS")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print("[ERROR] Model file not found:", args.model)
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Could not open camera index", args.camera)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Note: mediapipe==0.10.14 has no canned_gestures_classifier_options; use minimal config.
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recognizer = GestureRecognizer.create_from_options(options)

    like_count = 0
    last_like_time = 0.0
    effect = EffectState()

    process_period = 1.0 / PROCESS_FPS
    last_process_t = 0.0
    last_frame_time = time.time()
    show_fps_val = 0.0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        now = time.time()

        # throttle to PROCESS_FPS
        if (now - last_process_t) < process_period:
            effect.draw(frame_bgr, now)
            cv2.putText(frame_bgr, f"LIKE: {like_count}", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"LIKE: {like_count}", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            if args.show_fps:
                cv2.putText(frame_bgr, f"FPS: {show_fps_val:.1f}", (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame_bgr, f"FPS: {show_fps_val:.1f}", (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow("Like Counter (Thumb_Up)", frame_bgr)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('r'): like_count = 0
            continue
        last_process_t = now

        # preprocess & infer
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(now * 1000)
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        # fps
        dt = now - last_frame_time
        last_frame_time = now
        if dt > 0:
            show_fps_val = 1.0 / dt

        # parse top gesture for first hand
        like_detected = False
        hand_bbox: Optional[Tuple[int,int,int,int]] = None

        if result and result.gestures:
            top_for_first = result.gestures[0] if len(result.gestures) > 0 else []
            if top_for_first:
                top_cat = top_for_first[0]
                if top_cat.category_name == LIKE_LABEL and top_cat.score >= SCORE_THRESHOLD:
                    like_detected = True

        # bbox from landmarks (for effect center)
        if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
            h, w = frame_bgr.shape[:2]
            x, y, ww, hh = norm_rect_from_landmarks(result.hand_landmarks[0], w, h)
            hand_bbox = (x, y, ww, hh)
            if DRAW_LANDMARKS:
                for lm in result.hand_landmarks[0]:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame_bgr, (cx, cy), 2, (0, 180, 255), -1, cv2.LINE_AA)
                cv2.rectangle(frame_bgr, (x, y), (x+ww, y+hh), (0, 180, 255), 2, cv2.LINE_AA)

        # rising-edge with cooldown (3s)
        if like_detected and (now - last_like_time) >= COOLDOWN_SEC:
            like_count += 1
            last_like_time = now

            if (not args.no_sound):
                play_random_wav(args.audio_dir)

            # effect center
            if hand_bbox is not None:
                x, y, ww, hh = hand_bbox
                cx, cy = x + ww // 2, y + hh // 2
            else:
                h, w = frame_bgr.shape[:2]
                cx, cy = w // 2, h // 2
            effect.trigger(cx, cy, now, w, h)

        # draw overlays
        effect.draw(frame_bgr, now)
        cv2.putText(frame_bgr, f"LIKE: {like_count}", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"LIKE: {like_count}", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        if args.show_fps:
            cv2.putText(frame_bgr, f"FPS: {show_fps_val:.1f}", (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"FPS: {show_fps_val:.1f}", (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("Like Counter (Thumb_Up)", frame_bgr)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('r'): like_count = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
