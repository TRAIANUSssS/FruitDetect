from collections import Counter
import cv2
import numpy as np
from ultralytics import YOLO

# Предполагается, что fruit_classes определён где-то выше
fruit_classes = {
    47: "apple",
    # 46: "banana",
    # 49: "orange",
    # 51: 'carrot',
}

def process_video(video_path, output_video_path="output_video.mp4", output_image_path="output_frame.jpg"):
    """Обработка видео: сегментация, отслеживание по центроидам с устойчивостью и подсчет уникальных яблок."""
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

    # Подсчет уникальных яблок
    total_fruit_counts = Counter()
    last_annotated_frame = None
    last_fruit_counts = Counter()
    tracked_objects = []  # Список объектов: (id, x, y, fruit_name, age)
    next_id = 0  # Для назначения уникальных ID
    distance_threshold = 20  # Порог расстояния для нового яблока
    min_mask_area = 200  # Минимальная площадь маски
    max_mask_area = 5000  # Максимальная площадь маски
    max_age = 10  # Максимальное количество кадров без детекции
    confidence_threshold = 0.6  # Минимальная уверенность
    max_tracked_objects = 100  # Максимальное количество хранимых объектов

    # Морфологический фильтр для разделения масок
    kernel = np.ones((5, 5), np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Инференс модели
        results = model(frame, device="cuda", imgsz=1280, iou=0.5, conf=confidence_threshold)[0]

        # Подсчет яблок на текущем кадре
        frame_fruit_counts = Counter()
        current_centroids = []

        for box, mask in zip(results.boxes, results.masks or []):
            class_id = int(box.cls.cpu())
            if class_id in fruit_classes:
                fruit_name = fruit_classes[class_id]
                confidence = box.conf.cpu().item()
                if confidence < confidence_threshold:
                    continue

                # Получение маски и масштабирование
                mask_data = mask.data[0].cpu().numpy()
                mask_data = cv2.resize(mask_data, (width, height), interpolation=cv2.INTER_NEAREST)
                # Применяем эрозию для разделения слившихся масок
                mask_data = cv2.erode(mask_data, kernel, iterations=1)
                mask_coords = np.where(mask_data > 0)
                mask_area = len(mask_coords[0])
                if min_mask_area < mask_area < max_mask_area:
                    centroid_y = int(np.mean(mask_coords[0]))
                    centroid_x = int(np.mean(mask_coords[1]))
                    current_centroids.append((centroid_x, centroid_y, fruit_name))

                # Отрисовка маски
                mask_color = np.array([0, 0, 255], dtype=np.uint8)  # Красный для яблок
                frame[mask_data > 0] = frame[mask_data > 0] * 0.5 + mask_color * 0.5

                # Отрисовка метки
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                label = f"{fruit_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Отслеживание яблок
        new_tracked_objects = []
        for centroid_x, centroid_y, fruit_name in current_centroids:
            matched = False
            for obj in tracked_objects:
                obj_id, obj_x, obj_y, obj_name, obj_age = obj
                distance = np.sqrt((centroid_x - obj_x)**2 + (centroid_y - obj_y)**2)
                if distance < distance_threshold and fruit_name == obj_name:
                    # Обновляем существующий объект
                    new_tracked_objects.append((obj_id, centroid_x, centroid_y, fruit_name, 0))
                    matched = True
                    frame_fruit_counts[fruit_name] += 1
                    break
            if not matched:
                # Новый объект
                new_tracked_objects.append((next_id, centroid_x, centroid_y, fruit_name, 0))
                total_fruit_counts[fruit_name] += 1
                frame_fruit_counts[fruit_name] += 1
                next_id += 1

        # Обновляем возраст объектов, не найденных в этом кадре
        for obj in tracked_objects:
            obj_id, obj_x, obj_y, obj_name, obj_age = obj
            if obj not in [(o[0], o[1], o[2], o[3], o[4]) for o in new_tracked_objects]:
                if obj_age + 1 < max_age:
                    new_tracked_objects.append((obj_id, obj_x, obj_y, obj_name, obj_age + 1))

        # Ограничиваем количество объектов
        tracked_objects = new_tracked_objects[-max_tracked_objects:]

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
    print("Общее количество уникальных яблок за видео:", dict(total_fruit_counts))

    # Сохранение итогового изображения
    if last_annotated_frame is not None:
        results = model(last_annotated_frame, device="cuda", imgsz=1280, iou=0.5, conf=confidence_threshold)[0]
        for box, mask in zip(results.boxes, results.masks or []):
            class_id = int(box.cls.cpu())
            if class_id in fruit_classes:
                fruit_name = fruit_classes[class_id]
                confidence = box.conf.cpu().item()
                mask_data = mask.data[0].cpu().numpy()
                mask_data = cv2.resize(mask_data, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_data = cv2.erode(mask_data, kernel, iterations=1)
                mask_color = np.array([0, 0, 255], dtype=np.uint8)
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
        print("Яблоки не обнаружены в видео")

# Пример вызова
if __name__ == "__main__":
    model = YOLO("yolov8l-seg.pt")  # Используем более точную модель
    video_path = "3.mp4"
    process_video(video_path, output_video_path=f"out_{video_path}")