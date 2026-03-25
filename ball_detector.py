import cv2
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import os
import math

RED     = 0
YELLOW  = 1
GREEN   = 2
BLUE    = 3
PINK    = 4
WHITE   = 5
UNKNOWN = 6

COLOR_NAMES = {
    RED: "Red",
    YELLOW: "Yellow",
    GREEN: "Green",
    BLUE: "Blue",
    PINK: "Pink",
    WHITE: "White",
    UNKNOWN: "Unknown"
}

HUE_SECTORS = {
    RED:    [(0, 10), (170, 179)],
    YELLOW: [(11, 35)],
    GREEN:  [(36, 85)],
    BLUE:   [(86, 130)],
    PINK:   [(131, 169)]
}


def classify_pixel(h, s, v):
    if s < 35 and v > 160:
        return WHITE
    if v < 50 or s < 30:
        return UNKNOWN
    for color_code, ranges in HUE_SECTORS.items():
        for h_min, h_max in ranges:
            if h_min <= h <= h_max:
                return color_code
    return UNKNOWN


def ball_color_code(hsv_img, x, y, r):
    votes = []

    for angle in range(0, 360, 5):
        for radius in range(int(r * 0.25), int(r * 0.95)):
            px = int(x + radius * np.cos(np.deg2rad(angle)))
            py = int(y + radius * np.sin(np.deg2rad(angle)))

            if 0 <= px < hsv_img.shape[1] and 0 <= py < hsv_img.shape[0]:
                h, s, v = hsv_img[py, px]
                code = classify_pixel(h, s, v)
                if code != UNKNOWN:
                    votes.append(code)

    if not votes:
        return UNKNOWN, {}

    counter = Counter(votes)
    best_code = counter.most_common(1)[0][0]
    return best_code, counter


def detect_balls_and_colors(img):
    """
    img = ROI image (BGR numpy array)
    """

    output = img.copy()
    currentHeight, currentWidth = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=max((currentWidth*1.1)/1912,(currentHeight*1.1)/1080),
        minDist=max(int((currentWidth*30)/1912),int((currentHeight*30)/1080)),
        param1=max(int((currentWidth*100)/1912),int((currentHeight*100)/1080)),
        param2=max(int((currentWidth*18)/1912),int((currentHeight*18)/1080)),
        minRadius=max(int((currentWidth*26)/1912),int((currentHeight*26)/1080)),
        maxRadius=max(int((currentWidth*30)/1912),int((currentHeight*30)/1080)),
    )

    balls = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for cx, cy, cr in circles[0]:
            x = int(cx)
            y = int(cy)
            r = int(cr)

            code, hist = ball_color_code(hsv, x, y, r)
            balls.append((x, y, r, code, hist))

            name = COLOR_NAMES[code]

            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

            label = f"{name} [{code}]"
            cv2.putText(
                output,
                label,
                (x - 30, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

            total = sum(hist.values())
            if total > 0:
                confidence = int((hist[code] / total) * 100)
                cv2.putText(
                    output,
                    f"{confidence}%",
                    (x - r, y - r - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

    return output, balls





def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def balls_order_by_flow(balls, max_distance=50):
    """
    Returns a list of color codes representing the flow of the chain
    from head to tail, based on actual neighbor distances.

    balls: list of (x, y, r, code, hist)
    max_distance: max distance to consider two balls neighbors
    """
    if not balls:
        return []

    n = len(balls)
    used = [False] * n

    # Step 1: find neighbors for each ball
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        xi, yi, *_ = balls[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj, *_ = balls[j]
            if euclidean_distance((xi, yi), (xj, yj)) <= max_distance:
                neighbors[i].append(j)

    # Step 2: pick a head (ball with only one neighbor)
    head_idx = None
    for i, neigh in enumerate(neighbors):
        if len(neigh) == 1:
            head_idx = i
            break
    if head_idx is None:
        # fallback: pick the first ball if no endpoint found
        head_idx = 0

    # Step 3: follow the chain
    chain_order = [head_idx]
    used[head_idx] = True
    current = head_idx

    while len(chain_order) < n:
        min_dist = float('inf')
        next_idx = None
        for j in neighbors[current]:
            if not used[j]:
                dist = euclidean_distance((balls[current][0], balls[current][1]),
                                          (balls[j][0], balls[j][1]))
                if dist < min_dist:
                    min_dist = dist
                    next_idx = j
        if next_idx is None:
            # pick any unused ball if disconnected
            for j in range(n):
                if not used[j]:
                    next_idx = j
                    break
        chain_order.append(next_idx)
        used[next_idx] = True
        current = next_idx

    # Step 4: convert to color codes
    color_flow = [balls[i][3] for i in chain_order]
    return color_flow




def find_ball_chain(balls, max_distance=10):
    """
    Detects balls that form chains and returns only the longest chain.

    balls: list of (x, y, r, code, hist)
    max_distance: maximum distance between balls to consider them part of the same chain

    Returns: list of ball indices forming the longest chain (head first)
    """
    n = len(balls)
    used = [False] * n
    chains = []

    for i in range(n):
        if used[i]:
            continue

        chain = [i]
        used[i] = True
        current = i

        while True:
            # Find the closest ball of the same color not yet used
            min_dist = float('inf')
            next_idx = None
            for j in range(n):
                if used[j]:
                    continue
                if balls[j][3] != balls[current][3]:  # color mismatch
                    continue
                dist = euclidean_distance((balls[current][0], balls[current][1]),
                                          (balls[j][0], balls[j][1]))
                if dist < min_dist and dist <= max_distance:
                    min_dist = dist
                    next_idx = j
            if next_idx is None:
                break
            chain.append(next_idx)
            used[next_idx] = True
            current = next_idx

        chains.append(chain)

    if not chains:
        return []

    # Return only the longest chain
    longest_chain = max(chains, key=lambda c: len(c))
    return longest_chain





def get_color_from_code(code):
    if code == RED:
        return 'red'
    elif code == YELLOW:
        return 'yellow'
    elif code == GREEN:
        return 'green'
    elif code == BLUE:
        return 'blue'
    elif code == PINK:
        return 'pink'
    elif code == WHITE:
        return 'white'
    else:
        return 'gray'  # UNKNOWN





def save_balls_graph(balls, image_shape, chain_indices=None, filename="balls_graph.png"):
    """
    Save a graph of balls without showing it.
    Each ball is plotted with its color.
    Chain balls are highlighted.
    """
    height, width = image_shape
    fig, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # invert y-axis to match image coordinates
    ax.set_aspect('equal')

    for idx, ball in enumerate(balls):
        x, y, r, code, _ = ball
        color = get_color_from_code(code)
        if chain_indices and idx in chain_indices:
            ax.scatter(x, y, s=r*30, c=color, edgecolors='black', linewidths=1.5, zorder=2)
        else:
            ax.scatter(x, y, s=r*20, c=color, alpha=0.7, zorder=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Balls detected')

    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)  # important to prevent display









def consecutive_color_counter(balls, chain_indices):
    """
    balls: full balls list
    chain_indices: list of indices forming the chain
    """

    consecutive_balls = []

    if not chain_indices:
        return consecutive_balls, 0, -1

    for i in range(len(chain_indices)):
        idx = chain_indices[i]
        code = balls[idx][3]

        if i == 0:
            consecutive_balls.append({
                "code": code,
                "counter": 1
            })
        else:
            prev_idx = chain_indices[i - 1]
            prev_code = balls[prev_idx][3]
            prev_counter = consecutive_balls[i - 1]["counter"]

            if code == prev_code:
                consecutive_balls.append({
                    "code": code,
                    "counter": prev_counter + 1
                })
            else:
                consecutive_balls.append({
                    "code": code,
                    "counter": 1
                })

    max_counter = 0
    max_index = -1
    for i, item in enumerate(consecutive_balls):
        if item["counter"] > max_counter:
            max_counter = item["counter"]
            max_index = i

    return consecutive_balls, max_counter, max_index











#starting point detection
def get_edge_mask(h, w, thickness=10):
    mask = np.zeros((h, w), dtype=np.uint8)

    mask[:thickness, :] = 1               # top
    mask[-thickness:, :] = 1              # bottom
    mask[:, :thickness] = 1               # left
    mask[:, -thickness:] = 1              # right

    return mask

def compute_edge_motion(prev_gray, curr_gray, edge_mask):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Only keep motion on edges
    edge_motion = mag * edge_mask
    return edge_motion


def accumulate_edge_motion(edge_motion, motion_accumulator, decay=0.98):
    if motion_accumulator is None:
        return edge_motion.copy()

    motion_accumulator *= decay
    motion_accumulator += edge_motion
    return motion_accumulator


def detect_starting_point(motion_accumulator):
    y, x = np.unravel_index(
        np.argmax(motion_accumulator),
        motion_accumulator.shape
    )
    return int(x), int(y)


def visualize_starting_point(image, start_point, save_path):
    vis = image.copy()
    x, y = start_point

    cv2.drawMarker(
        vis,
        (x, y),
        (0, 0, 255),
        markerType=cv2.MARKER_STAR,
        markerSize=50,
        thickness=4
    )

    cv2.putText(
        vis,
        "START",
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.imwrite(save_path, vis)








# ============================
# BALL MATCHING
# ============================
def match_balls_by_distance(prev_balls, curr_balls, max_dist=30):
    matches = {}
    used = set()

    for i, (x1, y1, *_ ) in enumerate(prev_balls):
        best_j = None
        best_d = max_dist

        for j, (x2, y2, *_ ) in enumerate(curr_balls):
            if j in used:
                continue
            d = np.hypot(x2 - x1, y2 - y1)
            if d < best_d:
                best_d = d
                best_j = j

        if best_j is not None:
            matches[i] = best_j
            used.add(best_j)

    return matches

# ============================
# HEAD DETECTION FROM START
# ============================
def update_ball_tracks(prev_balls, curr_balls, matches, ball_tracks, next_id):
    for pi, ci in matches.items():
        x1, y1, _, code1, _ = prev_balls[pi]
        x2, y2, _, code2, _ = curr_balls[ci]

        d = np.hypot(x2 - x1, y2 - y1)

        # Find existing track
        found = False
        for tid, t in ball_tracks.items():
            if np.hypot(t["pos"][0] - x1, t["pos"][1] - y1) < 5:
                t["distance"] += d
                t["pos"] = (x2, y2)
                found = True
                break

        if not found:
            ball_tracks[next_id] = {
                "pos": (x2, y2),
                "distance": d,
                "color": code2
            }
            next_id += 1

    return next_id

def get_chain_by_motion(ball_tracks):
    ordered = sorted(
        ball_tracks.items(),
        key=lambda x: x[1]["distance"],
        reverse=True
    )
    return ordered


