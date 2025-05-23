
import sys
from ultralytics import YOLO
import gc
import torch

models_paths = [
    #'/home/marcio/projetos/lab/Yolo-DinoV2/yolo_dinov2_configs/yolo_dinov2_large.yaml',
    '/home/marcio/projetos/lab/Yolo-DinoV2/yolo_dinov2_configs/yolo_uni_large_multi_default.yaml',
    #'yolov8l-seg',
    #'/home/marcio/projetos/lab/Yolo-DinoV2/yolo_dinov2_configs/yolo_uni_large_multi.yaml',
    #'/home/marcio/projetos/lab/Yolo-DinoV2/yolo_dinov2_configs/yolo_uni_large_spatial.yaml',
    #'/home/marcio/projetos/lab/Yolo-DinoV2/yolo_dinov2_configs/yolo_uni_large_multi_spatial.yaml',
]

data_roboflow = ['/home/marcio/projetos/lab/timm-yolo/dataset/data.yaml', 224]
data_full = ['/home/marcio/Downloads/FULL_WSI_normal_sclerosis_yolo/data.yaml', 448]
data_part = ['/home/marcio/Downloads/WSI_normal_sclerosis_yolo/data.yaml', 448]

DATA, IMGZ = data_roboflow
EPOCHS = 100
for index, model_path  in enumerate(models_paths):
    model = YOLO(model_path)
    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=224,
        batch=4,      
        patience=10,
        save=True,
        device=0,
        lr0=0.0003,        
        lrf=0.01,
        optimizer="AdamW",
        weight_decay=0.0005,
        augment=True,
        dropout=0.1,
        warmup_epochs=3.0,  
        cos_lr=True          
    )

    # clean up
    del model

    gc.collect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()