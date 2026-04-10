import warnings
from pathlib import Path

import cv2

warnings.filterwarnings('ignore')

from ultralytics import RTDETR

# 可视化汇总图保存路径（相对本脚本目录）；多张图时仅第一张写入该文件，其余仍在 runs/detect/... 下
RESULT_PNG = Path(__file__).resolve().parent / 'result.png'

if __name__ == '__main__':
    model = RTDETR('D:\\Galaxy\\其他\\桌面\\RT-DETR\\runs\\train\\wool2\\weights\\best.pt')
    results = model.predict(
        source='testset',
        conf=0.25,
        imgsz=640,
        project='runs/detect',
        name='EFRT-DETR',
        save=True,
        # 羊毛尺寸/长宽比先验过滤（按数据分布调整 default.yaml 或此处传参）
        wool_prior_filter=True,
        wool_min_area_ratio=0.0001,
        wool_max_area_ratio=0.92,
        wool_min_aspect_ratio=1.0,
        wool_max_aspect_ratio=15.0,
    )
    # r.plot() 为 BGR ndarray，可直接存为 PNG
    if results:
        cv2.imwrite(str(RESULT_PNG), results[0].plot())
        print(f'Saved: {RESULT_PNG}')