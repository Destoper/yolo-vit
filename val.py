import sys
sys.path.append('/home/marcio/projetos/lab/Yolo-DinoV2/ultralytics')

from ultralytics import YOLO


model = YOLO("/home/marcio/projetos/lab/Yolo-DinoV2/runs/segment/train/weights/best.pt")
metrics = model.val()
print("Mean Average Precision for boxes:", metrics.box.map)
print("Mean Average Precision for masks:", metrics.seg.map)