import time
import os
import numpy as np
import cv2
from collections import Counter
import math
import random
import pyautogui
import keyboard
import threading
import sys

from roi import ScreenCap, ROIDetector
from detectShooter import run_motion
from ball_detector import (
    detect_balls_and_colors,
    find_ball_chain,
    consecutive_color_counter,
    save_balls_graph,
    get_edge_mask,
    compute_edge_motion,
    accumulate_edge_motion,
    detect_starting_point,
    visualize_starting_point,
    get_chain_by_motion,
    update_ball_tracks,
    match_balls_by_distance
)

# ---------------------------
# CONFIG
# ---------------------------
OUTPUT_FOLDER = "detecting balls live"
DETECT_INTERVAL = 1  # seconds
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

STOP_REQUESTED = False


def esc_listener():
    global STOP_REQUESTED
    keyboard.wait('esc')
    STOP_REQUESTED = True
    print("🛑 ESC pressed — terminating program...")


def resize_to_width(img, width):
    h, w = img.shape[:2]
    if w <= 0:
        return img
    s = width / float(w)
    return cv2.resize(img, (width, max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def assign_chain_priorities(balls, chain_indices):
    numbered_balls = []
    if chain_indices:
        chain_balls = [balls[i] for i in chain_indices]
        colors = [b[3] for b in chain_balls]
        color_counts = Counter(colors)
        for num, b in enumerate(chain_balls, 1):
            x, y, r, code, hist = b
            numbered_balls.append({
                "number": num,
                "x": x,
                "y": y,
                "r": r,
                "color_code": code,
                "color_name": COLOR_NAMES[code],
                "priority": color_counts[code]
            })
    return numbered_balls


def main():
    global STOP_REQUESTED

    listener = threading.Thread(target=esc_listener, daemon=True)
    listener.start()

    win = "ROI_ONLY"
    last_win_rect_screen = None
    motion_done = False
    last_detection_time = 0

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    roi_det = ROIDetector()

    fps = 0.0
    t_prev = time.perf_counter()
    MONITOR_INDEX = 1

    prev_balls = None
    ball_tracks = {}
    next_track_id = 0
    chain = []

    prev_gray = None
    edge_mask = None
    edge_motion_acc = None
    start_point = None
    start_locked = False
    start_frames_collected = 0
    START_DETECTION_FRAMES = 20

    roi_offset_x = 0
    roi_offset_y = 0

    with ScreenCap(MONITOR_INDEX) as cap:
        while not STOP_REQUESTED:
            frame, mon_off = cap.grab()

            t = time.perf_counter()
            dt = t - t_prev
            t_prev = t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            out = roi_det.update(frame, mon_off, last_win_rect_screen)

            if out["ok"]:
                x, y, w, h = out["bbox"]
                roi_offset_x = x
                roi_offset_y = y

                if not motion_done:
                    GAME_REGION = {"left": x, "top": y, "width": w, "height": h}
                    shooterx, shootery = run_motion(GAME_REGION)
                    motion_done = True

                roi = out["roi"].copy()
                cv2.rectangle(roi, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1), (0, 255, 0), 3)

                hud = f"FPS:{fps:.1f} | ROI:{roi.shape[1]}x{roi.shape[0]} | [ESC exit]"
                cv2.putText(roi, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(roi, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1)

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                if edge_mask is None:
                    h_, w_ = gray.shape
                    edge_mask = get_edge_mask(h_, w_, thickness=12)

                if not start_locked and prev_gray is not None:
                    edge_motion = compute_edge_motion(prev_gray, gray, edge_mask)
                    edge_motion_acc = accumulate_edge_motion(edge_motion, edge_motion_acc)
                    start_frames_collected += 1
                    if start_frames_collected >= START_DETECTION_FRAMES:
                        start_point = detect_starting_point(edge_motion_acc)
                        start_locked = True
                        visualize_starting_point(roi, start_point,
                                                 os.path.join(OUTPUT_FOLDER, "starting_point_locked.png"))

                prev_gray = gray

                now = time.time()
                if now - last_detection_time >= DETECT_INTERVAL:
                    annotated_roi, balls = detect_balls_and_colors(roi)

                    shooter_pos = (shooterx, shootery)
                    click_for_shooter_by_position(balls, shooter_pos, roi_offset_x, roi_offset_y)

                    if prev_balls is None:
                        prev_balls = balls
                        for b in balls:
                            x_, y_, r_, c_, _ = b
                            ball_tracks[next_track_id] = {"pos": (x_, y_), "distance": 0.0, "color": c_}
                            next_track_id += 1
                    else:
                        matches = match_balls_by_distance(prev_balls, balls)
                        next_track_id = update_ball_tracks(prev_balls, balls, matches, ball_tracks, next_track_id)
                        chain = get_chain_by_motion(ball_tracks)

                    prev_balls = balls
                    last_detection_time = now
                else:
                    annotated_roi = roi.copy()

                cv2.imshow(win, resize_to_width(annotated_roi, 900))

            else:
                vis = np.zeros((420, 720, 3), dtype=np.uint8)
                cv2.putText(vis, "Searching ROI ...", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                cv2.imshow(win, vis)

            try:
                last_win_rect_screen = cv2.getWindowImageRect(win)
            except Exception:
                last_win_rect_screen = None

            if cv2.waitKey(1) & 0xFF == 27:
                STOP_REQUESTED = True

    cv2.destroyAllWindows()
    sys.exit(0)


# ---------------------------
# GLOBAL VARIABLES
# ---------------------------
COLOR_NAMES = {
    0: "Red",
    1: "Yellow",
    2: "Green",
    3: "Blue",
    4: "Pink",
    5: "White",
    6: "Unknown"
}

def click_for_shooter_by_position(balls, shooter_pos, roi_offset_x, roi_offset_y):
    if not balls:
        return

    SHOOTER_RADIUS = 70
    FAR_DIST = 260
    CHAIN_LINK_DIST = 55

    sx, sy = shooter_pos

    # ------------------------------------------------------------
    # Detect shooter color (ONLY for color reference)
    shooter_color = None
    for b in balls:
        if math.hypot(b[0] - sx, b[1] - sy) <= SHOOTER_RADIUS:
            shooter_color = b[3]
            break

    if shooter_color is None:
        shooter_color = min(
            balls,
            key=lambda b: math.hypot(b[0] - sx, b[1] - sy)
        )[3]

    # ------------------------------------------------------------
    # HARD EXCLUDE shooter region completely
    balls = [b for b in balls if math.hypot(b[0] - sx, b[1] - sy) > SHOOTER_RADIUS]
    if not balls:
        return

    # ------------------------------------------------------------
    # Filter far & isolated balls
    valid_balls = []
    for b in balls:
        bx, by = b[0], b[1]
        dist = math.hypot(bx - sx, by - sy)

        has_neighbor = any(
            math.hypot(o[0] - bx, o[1] - by) <= 15
            for o in balls if o is not b
        )

        if dist <= FAR_DIST or has_neighbor:
            valid_balls.append(b)

    if not valid_balls:
        return

    # ------------------------------------------------------------
    # Group by color
    color_groups = {}
    for b in valid_balls:
        color_groups.setdefault(b[3], []).append(b)

    # ------------------------------------------------------------
    # Build chains
    all_chains = []

    for color, group in color_groups.items():
        if len(group) < 2:
            continue

        group.sort(key=lambda b: (b[1], b[0]))  # top-to-bottom
        current = [group[0]]

        for i in range(1, len(group)):
            prev, curr = current[-1], group[i]
            if (curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2 <= CHAIN_LINK_DIST ** 2:
                current.append(curr)
            else:
                if len(current) >= 2:
                    all_chains.append((color, current))
                current = [curr]

        if len(current) >= 2:
            all_chains.append((color, current))

    if not all_chains:
        return

    # ------------------------------------------------------------
    # Chain scoring (THIS IS THE FIX)
    def chain_score(color, chain):
        # Closest ball in chain (front-most)
        closest_ball = min(chain, key=lambda b: math.hypot(b[0] - sx, b[1] - sy))
        dist = math.hypot(closest_ball[0] - sx, closest_ball[1] - sy)

        same_color = (color == shooter_color)
        length = len(chain)

        # Higher score = higher priority
        score = (
            (1000 if same_color else 0) +   # absolute priority
            (length * 100) -                # longer chain = much higher priority
            dist                             # closer is better
        )
        return score, closest_ball

    best_score = -1e9
    target_ball = None

    for color, chain in all_chains:
        score, candidate = chain_score(color, chain)
        if score > best_score:
            best_score = score
            target_ball = candidate

    if target_ball is None:
        return

    # ------------------------------------------------------------
    # CLICK
    tx, ty = target_ball[0], target_ball[1]
    screen_x = tx + roi_offset_x
    screen_y = ty + roi_offset_y

    pyautogui.moveTo(screen_x, screen_y)
    pyautogui.click()


if __name__ == "__main__":
    main()
