import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# ========== 断点续训：只改这里 ==========
# True 时从下面 LAST_CKPT 接着训；False 时从零（yaml + 预训练）开始
RESUME = True
# 改成你中断的那次实验目录里的 weights/last.pt（与当时结构一致）
LAST_CKPT = r'D:\Galaxy\其他\桌面\RT-detr\runs\train\wool_small_obj\weights\last.pt'

if __name__ == '__main__':
    if RESUME:
        model = RTDETR(LAST_CKPT)
    else:
        model = RTDETR('ultralytics/cfg/models/rt-detr/wool_rtdetr.yaml')
        model.load('rtdetr-l.pt')

    model.train(
        data='demo.yaml',
        resume=RESUME,
        cache=True,
        # 小目标：在显存允许下尽量大；8G 可先试 800，有余量再改 896
        imgsz=800,
        epochs=200,
        batch=8,
        workers=0,
        device='0',
        project='runs/train',
        name='wool_small_obj',
        optimizer='AdamW',
        # 已有 COCO 预训练时用略稳的 lr，避免一上来破坏特征
        lr0=2e-4,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        # 本仓库里 warmup_epochs 表示「迭代步数」，见 engine/trainer.py
        warmup_epochs=1000,
        # 小目标：过强的随机缩放会把目标缩没；略减几何扰动
        scale=0.25,
        translate=0.1,
        degrees=5.0,
        shear=1.0,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.35,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        amp=False,
        pretrained=True,
        patience=30,
    )
