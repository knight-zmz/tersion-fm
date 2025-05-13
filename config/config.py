# config/config.py
"""
配置参数模块
"""

import os
from datetime import datetime

class Config:
    # 数据相关
    DATA_DIR = "path/to/pkl/files"  # 数据目录路径
    TRAIN_RATIO = 0.8  # 训练集比例
    VAL_RATIO = 0.1    # 验证集比例
    TEST_RATIO = 0.1   # 测试集比例
    
    # 模型相关
    TORSION_TYPES = [
    # 标准扭转角
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi",
    # 赝扭转角
    "eta", "theta", "eta'", "theta'",
    # 核糖构象角
    "v0", "v1", "v2", "v3", "v4"
    ]# 预测的扭转角类型
    
    HIDDEN_DIM = 256   # 回归头隐藏层维度
    DROPOUT = 0.1      # Dropout比例
    
    # 训练相关
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # 路径相关
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_DIR = "output"
    
    # 创建唯一的实验ID
    EXPERIMENT_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 硬件设置
    DEVICE = "cuda"  # 'cuda' or 'cpu'
    NUM_WORKERS = 4  # 数据加载器工作进程数
    
    def __init__(self):
        # 确保必要的目录存在
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # 当前实验的输出目录
        self.EXPERIMENT_DIR = os.path.join(self.OUTPUT_DIR, self.EXPERIMENT_ID)
        os.makedirs(self.EXPERIMENT_DIR, exist_ok=True)