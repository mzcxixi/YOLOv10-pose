from ultralytics import YOLOv10
model = YOLOv10('/home/mzc/YOLOv10-pose/runs/v10-pose/train18/weights/last.pt')
model.predict(source='/media/mzc/7ADE-3343/美国实验数据和数据集/dataset/keypointPose/valid/images', save=True)