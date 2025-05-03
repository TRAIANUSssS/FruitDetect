from ultralytics import YOLO

# Загрузка модели
model = YOLO("yolov8m-seg.pt")

# Вывод всех классов с их ID
_classes = ["apple",
"banana",
"orange",
"lemon",
"pear",
"grape",
"strawberry",
"pineapple",
"kiwi",]
print(model.names)

# for _class in _classes:
#     print(model.names[_class])