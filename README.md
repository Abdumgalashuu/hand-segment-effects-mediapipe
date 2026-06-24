# hand-segment-effects-mediapipe
Real-time MediaPipe hand tracking project that selects image processing effects using a 6-segment gesture wheel.
# Hand Segment Effects with MediaPipe

This project uses MediaPipe hand tracking and OpenCV to control real-time image processing effects with a 6-segment gesture wheel.

## Features

- Real-time hand/finger tracking with MediaPipe
- 6 segment gesture-based effect selector
- FPS display
- Live webcam processing
- OpenCV image effects

## Segments

| Segment | Effect |
|---|---|
| 1 | Canny |
| 2 | HSV Color Space |
| 3 | Cartoon Effect |
| 4 | Sepia |
| 5 | Pixelate / Mosaic |
| 6 | Sobel |

## Requirements

```bash
pip install opencv-python mediapipe numpy
