import cv2
import numpy as np
import mss
import pyautogui
from pynput import mouse
import time
import math
import threading

pyautogui.FAILSAFE = False

try:
    import ctypes
    use_win32 = True
except:
    use_win32 = False

PATCH_SIZE = 80
HOVER_DURATION = 0.25
CIRCLE_RADIUS = 74
ROTATIONS = 1.25

def run_motion(GAME_REGION):
    sct = mss.mss()
    click_x = GAME_REGION["left"] + GAME_REGION["width"] // 2
    click_y = GAME_REGION["top"] + GAME_REGION["height"] // 2

    pyautogui.moveTo(click_x, click_y, duration=0.3)
    time.sleep(0.3)

    # ----------------------------
    # SHARED FLAGS
    # ----------------------------
    hover_active = False
    start_hover_event = threading.Event()
    hover_done_event = threading.Event()

    # ----------------------------
    # WAIT FOR 2 MANUAL CLICKS
    # ----------------------------
    click_count = 0
    def on_click(x, y, button, pressed):
        nonlocal click_count
        if pressed:
            click_count += 1
            print(f"Click {click_count}")
            if click_count >= 2:
                start_hover_event.set()
                return False  # stop listener

    listener = mouse.Listener(on_click=on_click)
    listener.start()
    print("Waiting for 2 manual clicks to start hover...")
    start_hover_event.wait()

    # Optional 3-second pause before hover
    print("Starting hover in 3 seconds...")
    time.sleep(3)

    # ----------------------------
    # HOVER THREAD
    # ----------------------------
    def circular_hover():
        nonlocal hover_active
        hover_active = True
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= HOVER_DURATION:
                break

            angle = (elapsed / HOVER_DURATION) * (ROTATIONS * 2 * math.pi)
            target_x = click_x + int(CIRCLE_RADIUS * math.cos(angle))
            target_y = click_y + int(CIRCLE_RADIUS * math.sin(angle))

            if use_win32:
                ctypes.windll.user32.SetCursorPos(target_x, target_y)
            else:
                pyautogui.moveTo(target_x, target_y, duration=0, _pause=False)

            time.sleep(0.001)

        hover_active = False
        hover_done_event.set()

    # ----------------------------
    # MOTION DETECTION VARIABLES
    # ----------------------------
    motion_scores = {}
    last_gray = None
    last_frame = None

    hover_thread = threading.Thread(target=circular_hover, daemon=True)
    hover_thread.start()

    # ----------------------------
    # DETECTION LOOP: RUN ONLY DURING HOVER
    # ----------------------------
    frames_processed = False
    while not hover_done_event.is_set() or not frames_processed:
        if hover_active:
            frame = np.array(sct.grab(GAME_REGION))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            if last_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    last_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                for y in range(0, h - PATCH_SIZE, PATCH_SIZE // 2):
                    for x in range(0, w - PATCH_SIZE, PATCH_SIZE // 2):
                        patch_mag = mag[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                        motion_scores[(x, y)] = motion_scores.get((x, y), 0) + np.mean(patch_mag)

            last_gray = gray.copy()
            last_frame = frame.copy()
            frames_processed = True  # Ensure at least one frame is processed

        time.sleep(0.001)

    hover_thread.join()
    cv2.destroyAllWindows()

    # ----------------------------
    # SAVE RESULT IMMEDIATELY
    # ----------------------------
    if motion_scores and last_frame is not None:
        target_x, target_y = max(motion_scores, key=motion_scores.get)
        final = last_frame.copy()

        cv2.rectangle(final, (target_x, target_y),
                      (target_x + PATCH_SIZE, target_y + PATCH_SIZE),
                      (0, 0, 255), 3)
        cv2.putText(final, "FASTEST MOVING OBJECT",
                    (target_x, target_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imwrite("fastest_motion_detected.png", final)
        print("Fastest moving object detected and saved.")
    else:
        print("No motion detected.")

    return target_x, target_y

# ----------------------------
# EXAMPLE USAGE
# ----------------------------
if __name__ == "__main__":
    monitor = mss.mss().monitors[1]
    GAME_REGION = {
        "top": 0,
        "left": 0,
        "width": monitor["width"] // 2,
        "height": monitor["height"]
    }
    print("Prepare Zuma game window.")
    run_motion(GAME_REGION)
