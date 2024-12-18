from ultralytics import YOLOv10
model = YOLOv10('ultralytics/cfg/models/v10-pose/yolov10s-pose.yaml')
model.train(cfg='ultralytics/cfg/default.yaml',data='ultralytics/cfg/datasets/apple_pose.yaml', epochs=700, batch=8, imgsz=640)
