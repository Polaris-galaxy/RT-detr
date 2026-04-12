from pathlib import Path

from ultralytics import RTDETR

BATCH = 4

RESUME = False

LAST_PT = Path(__file__).resolve().parent / 'runs/train/wool_small_obj'

MODEL_YAML = 'ultralytics/cfg/models/rt-detr/wool_rtdetr.yaml'

PRETRAINED = 'rtdetr-l.pt'


def resolve_last_pt(path_or_dir: Path) -> Path:
    """解析为存在的 last.pt 路径。"""
    p = path_or_dir.expanduser().resolve()
    if p.is_file():
        return p
    for cand in (p / 'weights' / 'last.pt', p / 'last.pt'):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f'未找到 last.pt: 已尝试 {p}（文件）、{p / "weights" / "last.pt"}、{p / "last.pt"}'
    )


if __name__ == '__main__':
    if RESUME:
        ckpt = resolve_last_pt(LAST_PT)
        model = RTDETR(str(ckpt))
        model.train(
            data='demo.yaml',
            epochs=100,
            imgsz=640,
            batch=BATCH,
            resume=str(ckpt),
        )
    else:
        model = RTDETR(MODEL_YAML)
        model.load(PRETRAINED)
        model.train(
            data='demo.yaml',
            epochs=100,
            imgsz=640,
            batch=BATCH,
            resume=False,
        )
