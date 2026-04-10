import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    # 羊毛专项结构：4×AIFI + 解码器 3 层 / 查询 300 / ndp=4，损失见 wool_rtdetr.yaml 中 detr_loss_gain
    model = RTDETR('ultralytics/cfg/models/rt-detr/wool_rtdetr.yaml')
    # COCO 预训练：仅当权重与当前 YAML 结构一致时可 load（官方 rtdetr-l.pt 与自定义 DRBC 结构不兼容）。
    # 若有本仓库训练出的 checkpoint：model = RTDETR('path/to/last.pt')
    # model.load('path/to_compatible.pt')

    model.train(
        data='demo.yaml',
        cache=False,
        imgsz=640,  # 精度优先可改为 800（batch 酌减）
        epochs=100,
        batch=8,
        workers=0,
        device='0',
        project='runs/train',
        name='wool',
        optimizer='AdamW',
        lr0=1e-4,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0001,
        warmup_epochs=2000,  # 实为迭代步数（与 default.yaml 一致）
        # 多尺度随机缩放：RandomPerspective 中 scale=0.4 → 约 0.6~1.4 倍
        scale=0.4,
        translate=0.1,
        degrees=0.0,
        # 轻微光度畸变（相对 default 减弱）
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.25,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.3,  # 需分割多边形时才会实际生效；检测框数据下多为 no-op，保留配置以符合策略
        amp=False,  # RT-DETR 说明中建议关闭 AMP 以避免匹配/数值问题
        pretrained=False,
    )
    # TensorRT FP16 推理示例（动态 batch 与 half 在部分导出路径互斥，可按需二选一或走 ONNX→TRT）：
    # m = RTDETR('runs/train/wool/weights/best.pt')
    # m.export(format='engine', half=True, imgsz=640, batch=4)
    # m.export(format='onnx', dynamic=True, imgsz=640)
