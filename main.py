import cv2
import mediapipe as mp
import time
import math
import numpy as np
import os

# ====================== MEDIA PIPE TASKS ======================
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("❌ HATA: hand_landmarker.task dosyası bulunamadı!")
    exit()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,                      # iki el algılanıyor
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = HandLandmarker.create_from_options(options)
# ============================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # gecikmeyi azalt

# Çember parametreleri
center_x, center_y = 580, 360
radius = 50
num_segments = 6                       # 10 -> 6
angle_per_segment = 360 / num_segments

last_segment = -1
is_thumb_inside = False

SEGMENT_NAMES = ["Canny", "HSV", "Karikatur", "Sepia", "Pixelate", "Sobel"]

# Sepia matrisi bir kez hesaplanır
SEPIA_KERNEL = np.array([[0.272, 0.534, 0.131],
                         [0.349, 0.686, 0.168],
                         [0.393, 0.769, 0.189]])

# Segment çizgi uç noktalarını önceden hesapla (her frame trig hesabı yapma)
SEG_LINES = []
for i in range(num_segments):
    a = math.radians(i * angle_per_segment)
    SEG_LINES.append((int(center_x + radius * math.cos(a)),
                      int(center_y + radius * math.sin(a))))


def apply_effect(roi, seg):
    """Efektleri yalnızca ROI'ye uygula. Hepsi tek geçiş, blur/overlay yok."""
    if seg == 0:  # Canny
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif seg == 1:  # HSV renk uzayı
        return cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    elif seg == 2:  # Karikatür
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(roi, 9, 250, 250)
        return cv2.bitwise_and(color, color, mask=edges)

    elif seg == 3:  # Sepia
        sep = cv2.transform(roi, SEPIA_KERNEL)
        return np.clip(sep, 0, 255).astype(np.uint8)

    elif seg == 4:  # Pixelate / Mozaik
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        small = cv2.resize(roi, (max(1, w // 12), max(1, h // 12)),
                           interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    elif seg == 5:  # Sobel
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        sy = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        s = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5,
                                                cv2.convertScaleAbs(sy), 0.5, 0))
        return cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)

    return roi


def draw_dial(frame, active_seg, highlight):
    """Çemberi hafifçe çiz. Mask/blur/addWeighted yok."""
    cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 2)
    for p in SEG_LINES:
        cv2.line(frame, (center_x, center_y), p, (255, 255, 255), 1)
    if active_seg in range(num_segments):
        sa = active_seg * angle_per_segment
        color = (0, 0, 255) if highlight else (0, 120, 255)
        cv2.ellipse(frame, (center_x, center_y), (radius, radius),
                    sa, 0, angle_per_segment, color, -1)


prev_t = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    is_thumb_inside = False
    points = []
    thumb_x = thumb_y = None
    h, w = frame.shape[:2]

    if detection_result.hand_landmarks:
        # Tüm ellerin baş (4) ve işaret (8) parmak uçlarını topla
        for hand in detection_result.hand_landmarks:
            ix = int(hand[8].x * w)
            iy = int(hand[8].y * h)
            tx = int(hand[4].x * w)
            ty = int(hand[4].y * h)
            points.extend([(tx, ty), (ix, iy)])

            # Sağ üst köşe ile çıkış (herhangi bir elin işaret parmağı)
            if ix > w - 100 and iy < 100:
                points = None  # çıkış sinyali
                break

            # Segment seçimi: baş parmak çemberin içindeyse
            dist = math.hypot(tx - center_x, ty - center_y)
            if dist <= radius * 1.2:
                is_thumb_inside = True
                angle = (math.degrees(math.atan2(ty - center_y,
                                                 tx - center_x)) + 360) % 360
                seg = int(angle // angle_per_segment)
                if seg >= num_segments:
                    seg = num_segments - 1
                last_segment = seg

        if points is None:
            break

    # Efekti ROI'ye uygula (tam ekran değil!)
    if points and last_segment in range(num_segments):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min = max(0, min(xs) - 20)
        x_max = min(w, max(xs) + 20)
        y_min = max(0, min(ys) - 20)
        y_max = min(h, max(ys) + 20)
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            frame[y_min:y_max, x_min:x_max] = apply_effect(roi, last_segment)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (200, 200, 200), 1)

    # Dial çizimi
    if last_segment != -1:
        draw_dial(frame, last_segment, is_thumb_inside)

    # FPS
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev_t))
    prev_t = now

    label = SEGMENT_NAMES[last_segment] if last_segment in range(num_segments) else "Yok"
    cv2.putText(frame, f"Segment: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Hand Segment Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
del landmarker