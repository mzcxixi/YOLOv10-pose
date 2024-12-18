from ultralytics import YOLOv10
model = YOLOv10('runs/pose/train18/weights/last.pt')
model.val(batch=16)
