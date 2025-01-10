import cv2
from ultralytics import YOLO
import numpy as np
from sort import Sort

def get_bright_color():
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),  # Red, Green, Blue
        (255, 255, 0), (0, 255, 255), (255, 0, 255),  # Yellow, Cyan, Magenta
        (255, 128, 0), (128, 0, 255), (255, 0, 128),  # Orange, Purple, Pink
        (128, 128, 0), (0, 128, 128), (128, 0, 128)  # Olive, Teal, Maroon
    ]
    return colors[np.random.choice(len(colors))]

def track_ball(model, file_location, confidence):
    cap = cv2.VideoCapture(file_location)
    if not cap.isOpened():
        print("Errore nell'apertura del file video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "processed_videos/output.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trajectory = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, classes=[32], conf=confidence)
        for result in results:
            for bbox in result.boxes:
                if bbox.cls == 32:
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    trajectory.append((center_x, center_y))
                    if len(trajectory) > 30:
                        trajectory.pop(0)
                    break

        for point in trajectory:
            cv2.circle(frame, point, 6, (0, 255, 255), -1)  # Linea di traiettoria più spessa

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Predizione completata e risultato salvato come output.mp4.")

def track_players(model, file_location, confidence):
    cap = cv2.VideoCapture(file_location)
    if not cap.isOpened():
        print("Errore nell'apertura del file video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "processed_videos/output.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = Sort()
    colors = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, classes=[0], conf=confidence)

        detections = []
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                score = float(bbox.conf[0])
                detections.append([x1, y1, x2, y2, score])

        trackers = tracker.update(np.array(detections))

        for trk in trackers:
            x1, y1, x2, y2, obj_id = map(int, trk)
            if obj_id not in colors:
                colors[obj_id] = get_bright_color()
            color = colors[obj_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 6)  # Bounding box più spesso
            cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 4)  # Testo più grande e spesso

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Predizione completata e risultato salvato come output.mp4.")
