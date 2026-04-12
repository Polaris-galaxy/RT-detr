import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    # 推理/跟踪优先使用验证集最优权重 best.pt；last.pt 为最后一轮，末期若过拟合可能较差
    model = RTDETR(r'runs\train\wool_small_obj\weights\best.pt')
    model.track(source=r'D:\Galaxy\其他\桌面\output_video53.avi',
                project='runs/track',
                name='exp',
                save=True
                )