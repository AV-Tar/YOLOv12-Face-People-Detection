import cv2
import os
from ultralytics import YOLO

# Загружаем модель
model = YOLO('yolov12n-face.pt').to('cpu')

# Создаем папку для скриншотов
save_dir = 'screens'

## создание папки, если ее нет
# os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
i = 0

while True:
    i += 1

    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    annotated_frame = results[0].plot()

    # --- ФИЛЬТР 1: сохраняем только каждый 10-й кадр ---
    if i % 10 == 0:
        filename = os.path.join(save_dir, f'{i}.png')

        # --- ФИЛЬТР 2: сохранять только если найдены лица ---
        if len(results[0].boxes) > 0:  # раскомментируй, если нужно сохранять только кадры с лицами
            cv2.imwrite(filename, annotated_frame)

    cv2.imshow("YOLOv12 Face Detection", annotated_frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

