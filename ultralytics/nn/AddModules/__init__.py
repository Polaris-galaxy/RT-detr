# 勿 `import *`：SimAM.py 内的 Conv 会覆盖 ultralytics.nn.modules.Conv，导致 YAML 中 Conv 解析错误
from .SimAM import SimAM, HGBlock_SimAM
from .SPDConv import SPDConv