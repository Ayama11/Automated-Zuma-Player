
import time
import numpy as np
import cv2


class ScreenCap:
    def __init__(self, monitor_index=1):
        self.monitor_index = monitor_index
        self.sct = None
        self.mon = None
        self.use_mss = False

    def __enter__(self):
        try:
            import mss
            self.sct = mss.mss()
            mons = self.sct.monitors
            if self.monitor_index < 1 or self.monitor_index >= len(mons):
                self.monitor_index = 1
            self.mon = mons[self.monitor_index]
            self.use_mss = True
        except Exception:
            self.use_mss = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass

    def grab(self):
        if self.use_mss:
            img = np.array(self.sct.grab(self.mon), dtype=np.uint8)  # BGRA
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return frame, (int(self.mon["left"]), int(self.mon["top"]))
        else:
            import pyautogui
            img = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return frame, (0, 0)


def _clip_bbox(b, W, H):
    x, y, w, h = b
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def _ema_bbox(prev, cur, a):
    if prev is None:
        return tuple(map(float, cur))
    px, py, pw, ph = prev
    cx, cy, cw, ch = cur
    return (
        a * cx + (1 - a) * px,
        a * cy + (1 - a) * py,
        a * cw + (1 - a) * pw,
        a * ch + (1 - a) * ph,
    )


def _auto_canny(gray, sigma):
    v = np.median(gray)
    t1 = int(max(0, (1.0 - sigma) * v))
    t2 = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, t1, t2)


def _lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _rect_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / (union + 1e-6))


class ROIDetector:
    def __init__(self):
        # parameters
        self.REDETECT_EVERY = 5
        self.EMA_ALPHA = 0.35
        self.SIGMA = 0.33

        self.SAT_PERC = 80
        self.SAT_MIN  = 75
        self.VAL_MIN  = 40

        self.CLOSE_K = 9
        self.OPEN_K  = 5
        self.K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.CLOSE_K, self.CLOSE_K))
        self.K_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.OPEN_K, self.OPEN_K))

        self.MIN_AREA_RATIO = 0.04
        self.MAX_AREA_RATIO = 0.85

        self.BOTTOM_MIN_RATIO = 0.07
        self.BOTTOM_MAX_RATIO = 0.45

        self.MIN_VALID = 0.52
        self.SELF_IOU_REJECT = 0.25

        # state
        self.prev_bbox = None
        self.score = 0.0
        self.frame_i = 0

    def reset(self):
        self.prev_bbox = None
        self.score = 0.0
        self.frame_i = 0

    def _hsv_mask(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        thr = int(max(self.SAT_MIN, np.percentile(s, self.SAT_PERC)))
        mask = ((s >= thr) & (v >= self.VAL_MIN)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.K_CLOSE, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.K_OPEN,  iterations=1)
        return mask

    def _find_candidates(self, mask, W, H):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            if cv2.contourArea(c) <= 0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            r = (w * h) / float(W * H)
            if self.MIN_AREA_RATIO <= r <= self.MAX_AREA_RATIO:
                out.append((x, y, w, h))
        return out

    def _bottom_cut(self, gray_patch):
        h = gray_patch.shape[0]
        edges = _auto_canny(cv2.GaussianBlur(gray_patch, (5, 5), 0), self.SIGMA)

        row_edge = (edges > 0).mean(axis=1)
        row_var  = gray_patch.var(axis=1)

        top_end = max(1, int(0.70 * h))
        base_edge = np.median(row_edge[:top_end]) + 1e-6
        base_var  = np.median(row_var[:top_end])  + 1e-6

        edge_thr = 0.40 * base_edge
        var_thr  = 0.40 * base_var

        cnt = 0
        for i in range(h - 1, -1, -1):
            if (row_edge[i] < edge_thr) and (row_var[i] < var_thr):
                cnt += 1
            else:
                break

        min_px = int(self.BOTTOM_MIN_RATIO * h)
        max_px = int(self.BOTTOM_MAX_RATIO * h)
        return min(cnt, max_px) if cnt >= min_px else 0

    def _refine_bottom(self, gray_full, bbox):
        H, W = gray_full.shape[:2]
        x, y, w, h = _clip_bbox(bbox, W, H)
        patch = gray_full[y:y+h, x:x+w]
        cut = self._bottom_cut(patch)
        return _clip_bbox((x, y, w, max(80, h - cut)), W, H)

    def _score_bbox(self, gray_full, sat_mask, bbox):
        H, W = gray_full.shape[:2]
        x, y, w, h = _clip_bbox(bbox, W, H)

        roi = gray_full[y:y+h, x:x+w]
        edges = _auto_canny(cv2.GaussianBlur(roi, (5, 5), 0), self.SIGMA)

        edge_density = float((edges > 0).mean())
        tex = _lap_var(roi)

        m = sat_mask[y:y+h, x:x+w]
        sat_density = float((m > 0).mean())

        edge_s = np.clip((edge_density - 0.018) / 0.10, 0, 1)
        tex_s  = np.clip((tex - 45.0) / 650.0, 0, 1)
        sat_s  = np.clip((sat_density - 0.03) / 0.50, 0, 1)

        return float(0.45 * tex_s + 0.25 * edge_s + 0.30 * sat_s)

    def _select_best(self, frame_bgr, gray, self_rect):
        H, W = frame_bgr.shape[:2]
        mask = self._hsv_mask(frame_bgr)
        candidates = self._find_candidates(mask, W, H)

        best_bbox = None
        best_score = -1.0

        for b in candidates:
            if self_rect is not None and _rect_iou(b, self_rect) > self.SELF_IOU_REJECT:
                continue
            rb = self._refine_bottom(gray, b)
            s = self._score_bbox(gray, mask, rb)
            if s > best_score:
                best_score = s
                best_bbox = rb

        if best_bbox is None or best_score < self.MIN_VALID:
            return None, 0.0
        return best_bbox, best_score

    def update(self, frame_bgr, mon_off, self_win_rect_screen):
        self.frame_i += 1
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # self rect relative to captured frame (mirror rejection)
        self_rect = None
        if self_win_rect_screen is not None:
            wx, wy, ww, wh = self_win_rect_screen
            mx, my = mon_off
            self_rect = (wx - mx, wy - my, ww, wh)

        if self.prev_bbox is None or (self.frame_i % self.REDETECT_EVERY == 0):
            best, best_s = self._select_best(frame_bgr, gray, self_rect)
            if best is not None:
                self.prev_bbox = _ema_bbox(self.prev_bbox, best, self.EMA_ALPHA)
                self.score = best_s

        if self.prev_bbox is None:
            return {"ok": False, "roi": None, "bbox": None, "score": 0.0}

        x, y, w, h = _clip_bbox(self.prev_bbox, W, H)
        roi = frame_bgr[y:y+h, x:x+w].copy()
        return {"ok": True, "roi": roi, "bbox": (x, y, w, h), "score": float(self.score)}









