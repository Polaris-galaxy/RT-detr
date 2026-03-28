import warnings
warnings.filterwarnings('ignore')

from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('D:\\Galaxy\\其他\\桌面\\RT-DETR\\RT-DETR\\runs\\train\\wool\\weights\\best.pt')
    model.predict(source='testset',
                  conf=0.25,
                  project='runs/detect',
                  name='EFRT-DETR',
                  save=True)