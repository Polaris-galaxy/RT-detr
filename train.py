import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/DRBC.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='demo.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=0,
                device='0',
                #resume='EFRT-DETR-NEW.yaml3', # last.pt path
                project='runs/train',
                name='wool',
                )