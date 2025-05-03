from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from collections import Counter
import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from pathlib import Path
import shutil
import subprocess

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Кастомный StaticFiles для явного Content-Type
class CustomStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if path.endswith(".mp4"):
            response.headers["Content-Type"] = "video/mp4"
            response.headers["Accept-Ranges"] = "bytes"
        return response

# Подключение папки static
app.mount("/static", CustomStaticFiles(directory="static"), name="static")

# Папка для сохранения результатов
os.makedirs("static", exist_ok=True)

# Загрузка модели
model = YOLO("yolov8l-seg.pt")

# Маппинг классов
fruit_classes = {
    47: "apple",
    46: "banana",
    49: "orange",
}

def check_ffmpeg():
    """Проверка доступности ffmpeg."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        print(f"FFmpeg version: {result.stdout.splitlines()[0]}")
        return True
    except FileNotFoundError:
        print("Ошибка: FFmpeg не найден. Убедитесь, что FFmpeg установлен и добавлен в PATH.")
        return False

def process_file(file_path, output_path, selected_categories):
    """Обработка изображения или видео с фильтрацией по выбранным категориям."""
    print(f"Processing file: {file_path}")
    is_image = file_path.lower().endswith((".jpg", ".jpeg", ".png"))

    if is_image:
        frame = cv2.imread(file_path)
        if frame is None:
            raise ValueError("Не удалось загрузить изображение")

        results = model(frame, device="cuda", imgsz=1280, iou=0.5, conf=0.6)[0]
        fruit_counts = Counter()
        kernel = np.ones((5, 5), np.uint8)

        for box, mask in zip(results.boxes, results.masks or []):
            class_id = int(box.cls.cpu())
            fruit_name = fruit_classes.get(class_id)
            if fruit_name not in selected_categories:
                continue

            confidence = box.conf.cpu().item()
            if confidence < 0.6:
                continue

            mask_data = mask.data[0].cpu().numpy()
            mask_data = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_data = cv2.erode(mask_data, kernel, iterations=1)
            mask_coords = np.where(mask_data > 0)
            mask_area = len(mask_coords[0])
            if 200 < mask_area < 5000:
                fruit_counts[fruit_name] += 1

            mask_color = np.array([0, 0, 255], dtype=np.uint8) if fruit_name == "apple" else \
                         np.array([0, 255, 255], dtype=np.uint8) if fruit_name == "banana" else \
                         np.array([0, 165, 255], dtype=np.uint8)
            frame[mask_data > 0] = frame[mask_data > 0] * 0.5 + mask_color * 0.5

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
            label = f"{fruit_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        y_offset = 30
        for fruit, count in fruit_counts.items():
            text = f"{fruit}: {count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        print(f"Saving processed image to: {output_path}")
        cv2.imwrite(output_path, frame)
        if not os.path.exists(output_path):
            raise ValueError("Не удалось сохранить обработанное изображение")
        return fruit_counts

    else:
        print("Video processing started")
        if not check_ffmpeg():
            raise ValueError("FFmpeg не доступен")

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("Не удалось открыть видео")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Используем временный файл для записи
        temp_output_path = output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            raise ValueError("Не удалось инициализировать VideoWriter")

        total_fruit_counts = Counter()
        tracked_objects = []
        next_id = 0
        distance_threshold = 20
        min_mask_area = 200
        max_mask_area = 5000
        max_age = 10
        max_tracked_objects = 100
        kernel = np.ones((5, 5), np.uint8)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, device="cuda", imgsz=1280, iou=0.5, conf=0.6)[0]
            frame_fruit_counts = Counter()
            current_centroids = []

            for box, mask in zip(results.boxes, results.masks or []):
                class_id = int(box.cls.cpu())
                fruit_name = fruit_classes.get(class_id)
                if fruit_name not in selected_categories:
                    continue

                confidence = box.conf.cpu().item()
                if confidence < 0.6:
                    continue

                mask_data = mask.data[0].cpu().numpy()
                mask_data = cv2.resize(mask_data, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_data = cv2.erode(mask_data, kernel, iterations=1)
                mask_coords = np.where(mask_data > 0)
                mask_area = len(mask_coords[0])
                if min_mask_area < mask_area < max_mask_area:
                    centroid_y = int(np.mean(mask_coords[0]))
                    centroid_x = int(np.mean(mask_coords[1]))
                    current_centroids.append((centroid_x, centroid_y, fruit_name))

                mask_color = np.array([0, 0, 255], dtype=np.uint8) if fruit_name == "apple" else \
                             np.array([0, 255, 255], dtype=np.uint8) if fruit_name == "banana" else \
                             np.array([0, 165, 255], dtype=np.uint8)
                frame[mask_data > 0] = frame[mask_data > 0] * 0.5 + mask_color * 0.5

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                label = f"{fruit_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            new_tracked_objects = []
            for centroid_x, centroid_y, fruit_name in current_centroids:
                matched = False
                for obj in tracked_objects:
                    obj_id, obj_x, obj_y, obj_name, obj_age = obj
                    distance = np.sqrt((centroid_x - obj_x)**2 + (centroid_y - obj_y)**2)
                    if distance < distance_threshold and fruit_name == obj_name:
                        new_tracked_objects.append((obj_id, centroid_x, centroid_y, fruit_name, 0))
                        matched = True
                        frame_fruit_counts[fruit_name] += 1
                        break
                if not matched:
                    new_tracked_objects.append((next_id, centroid_x, centroid_y, fruit_name, 0))
                    total_fruit_counts[fruit_name] += 1
                    frame_fruit_counts[fruit_name] += 1
                    next_id += 1

            for obj in tracked_objects:
                obj_id, obj_x, obj_y, obj_name, obj_age = obj
                if obj not in [(o[0], o[1], o[2], o[3], o[4]) for o in new_tracked_objects]:
                    if obj_age + 1 < max_age:
                        new_tracked_objects.append((obj_id, obj_x, obj_y, obj_name, obj_age + 1))

            tracked_objects = new_tracked_objects[-max_tracked_objects:]

            y_offset = 30
            for fruit, count in frame_fruit_counts.items():
                text = f"{fruit}: {count}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        print(f"Processed {frame_count} frames")

        # Проверяем, что временный файл создан
        if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
            raise ValueError("Не удалось сохранить временное видео")

        # Конвертируем видео в формат, совместимый с браузерами
        final_output_path = output_path
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{temp_output_path}" -c:v libx264 -preset fast -crf 22 '
            f'-c:a aac -b:a 128k -movflags +faststart -f mp4 "{final_output_path}"'
        )
        print(f"Running ffmpeg command: {ffmpeg_cmd}")
        ffmpeg_result = os.system(ffmpeg_cmd)
        if ffmpeg_result != 0:
            raise ValueError(f"FFmpeg failed with exit code {ffmpeg_result}")

        # Проверяем, что финальный файл создан
        if not os.path.exists(final_output_path) or os.path.getsize(final_output_path) == 0:
            raise ValueError("Не удалось конвертировать видео в H.264")

        # Проверяем воспроизведение
        cap = cv2.VideoCapture(final_output_path)
        if not cap.isOpened():
            raise ValueError("Созданное видео не может быть открыто")
        cap.release()

        os.remove(temp_output_path)
        print(f"Saving processed video to: {final_output_path}, size: {os.path.getsize(final_output_path)} bytes")
        return total_fruit_counts

@app.post("/process")
async def process(file: UploadFile = File(...), categories: str = Form(...)):
    """Обработка загруженного файла с учетом выбранных категорий."""
    print(f"Received file: {file.filename}, categories: {categories}")
    try:
        input_path = f"static/{file.filename}"
        print(f"Saving input file to: {input_path}")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        selected_categories = json.loads(categories)
        print(f"Selected categories: {selected_categories}")
        output_filename = f"proc{file.filename}"  # Убрали нижнее подчеркивание
        output_path = f"static/{output_filename}"

        fruit_counts = process_file(input_path, output_path, selected_categories)
        print(f"Fruit counts: {fruit_counts}")
        os.remove(input_path)

        print(f"Returning output_url: /static/{output_filename}")
        return JSONResponse(content={
            "output_url": f"/static/{output_filename}",
            "fruit_counts": dict(fruit_counts)
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
