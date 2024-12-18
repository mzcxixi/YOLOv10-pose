from ultralytics import YOLOv10
model = YOLOv10('/home/mzc/YOLOv10-pose/runs/v10-pose/train18/weights/last.pt')
model.predict(source='valid/images', save=True)
