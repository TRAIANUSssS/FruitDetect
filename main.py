import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import os
from norfair import Tracker, Detection
from scipy.spatial import distance as dist

# Загрузка предобученной модели YOLOv8
model = YOLO("yolov8m-seg.pt")  # Можно заменить на yolov8m-seg.pt для сегментации

# Список классов фруктов из COCO
fruit_classes = {
    47: "apple",
    # 46: "banana",
    # 49: "orange",
    # 51: 'carrot',
}

def process_image(image_path, output_path="output_image.jpg"):
    """Обработка изображения: детектирование и подсчет фруктов."""
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return

    # Инференс модели
    results = model(img, device="cuda")[0]  # Используем GPU

    # Подсчет фруктов
    fruit_counts = Counter()
    for box in results.boxes:
        class_id = int(box.cls.cpu())  # Переносим данные на CPU
        if class_id in fruit_classes:
            fruit_name = fruit_classes[class_id]
            fruit_counts[fruit_name] += 1

            # Отрисовка bounding box и метки
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
            confidence = box.conf.cpu().item()
            label = f"{fruit_name} {confidence:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Вывод статистики на изображение
    y_offset = 30
    for fruit, count in fruit_counts.items():
        text = f"{fruit}: {count}"
        cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    # Сохранение результата
    cv2.imwrite(output_path, img)
    print(f"Результат сохранен в {output_path}")
    print("Подсчет фруктов:", dict(fruit_counts))

def process_video(video_path, output_video_path="output_video.mp4", output_image_path="output_frame.jpg"):
    """Обработка видео: сегментация, отслеживание с norfair и подсчет уникальных фруктов."""
    # Открытие видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return

    # Параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Настройка записи выходного видео
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Инициализация трекера norfair
    def distance_function(detected_objects, points):
        # Извлекаем координаты из detected_objects (TrackedObject)
        obj_points = np.array([obj.last_detection.points[0] for obj in detected_objects if obj.last_detection.points is not None])
        if obj_points.size == 0 or points.size == 0:
            return np.array([])  # Возвращаем пустой массив, если нет объектов
        return dist.cdist(obj_points, points)

    tracker = Tracker(distance_function=distance_function, distance_threshold=50)

    # Подсчет уникальных фруктов
    total_fruit_counts = Counter()
    last_annotated_frame = None
    last_fruit_counts = Counter()
    tracked_ids = set()
    min_mask_area = 200

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Инференс модели
        results = model(frame, device="cuda")[0]

        # Подсчет фруктов на текущем кадре
        frame_fruit_counts = Counter()
        detections = []

        for box, mask in zip(results.boxes, results.masks or []):
            class_id = int(box.cls.cpu())
            if class_id in fruit_classes:
                fruit_name = fruit_classes[class_id]
                confidence = box.conf.cpu().item()
                if confidence < 0.7:  # Игнорировать низкую уверенность
                    continue

                # Получение маски и масштабирование
                mask_data = mask.data[0].cpu().numpy()
                mask_data = cv2.resize(mask_data, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_coords = np.where(mask_data > 0)
                mask_area = len(mask_coords[0])
                if mask_area > min_mask_area:
                    centroid_y = int(np.mean(mask_coords[0]))
                    centroid_x = int(np.mean(mask_coords[1]))
                    # Используем одномерный массив для points
                    detections.append(Detection(points=np.array([centroid_x, centroid_y]), data=fruit_name))

                # Отрисовка маски
                mask_color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                frame[mask_data > 0] = frame[mask_data > 0] * 0.5 + mask_color * 0.5

                # Отрисовка метки
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                label = f"{fruit_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Отслеживание с norfair
        if detections:  # Проверяем, что detections не пустой
            tracked_objects = tracker.update(detections)
            for obj in tracked_objects:
                fruit_name = obj.data
                obj_id = obj.id
                if obj_id not in tracked_ids:
                    tracked_ids.add(obj_id)
                    total_fruit_counts[fruit_name] += 1
                    frame_fruit_counts[fruit_name] += 1
        else:
            tracked_objects = []  # Пропускаем трекинг, если нет detections

        # Вывод статистики на кадр
        y_offset = 30
        for fruit, count in frame_fruit_counts.items():
            text = f"{fruit}: {count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        # Запись кадра
        out.write(frame)

        # Сохранение последнего кадра
        if frame_fruit_counts:
            last_annotated_frame = frame.copy()
            last_fruit_counts = frame_fruit_counts

    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Сохранение итогового видео
    print(f"Аннотированное видео сохранено в {output_video_path}")
    print("Общее количество уникальных фруктов за видео:", dict(total_fruit_counts))

    # Сохранение итогового изображения
    if last_annotated_frame is not None:
        results = model(last_annotated_frame, device="cuda")[0]
        for box, mask in zip(results.boxes, results.masks or []):
            class_id = int(box.cls.cpu())
            if class_id in fruit_classes:
                fruit_name = fruit_classes[class_id]
                confidence = box.conf.cpu().item()
                mask_data = mask.data[0].cpu().numpy()
                mask_data = cv2.resize(mask_data, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                last_annotated_frame[mask_data > 0] = last_annotated_frame[mask_data > 0] * 0.5 + mask_color * 0.5
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                label = f"{fruit_name} {confidence:.2f}"
                cv2.putText(last_annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        y_offset = 30
        for fruit, count in total_fruit_counts.items():
            text = f"{fruit}: {count}"
            cv2.putText(last_annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        cv2.imwrite(output_image_path, last_annotated_frame)
        print(f"Итоговое изображение сохранено в {output_image_path}")
    else:
        print("Фрукты не обнаружены в видео")

if __name__ == "__main__":
    # Для изображения
    # image_path = "1.jpg"  # Укажите путь к вашему изображению
    # if os.path.exists(image_path):
    #     process_image(image_path)

    # Для видео
    for i in range (1,3):
        video_path = f"{i}.mp4"  # Укажите путь к вашему видео
        if os.path.exists(video_path):
            process_video(video_path, output_video_path=f"out_{video_path}")