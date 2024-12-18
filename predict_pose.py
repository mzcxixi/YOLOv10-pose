from ultralytics import YOLOv10
model = YOLOv10('runs/v10-pose/train18/weights/last.pt')
model.predict(source='valid/images', save=True)
