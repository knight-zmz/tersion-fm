# models/torsion_predictor.py
"""
RNA扭转角预测模型定义

该模型基于预训练的RNA-FM模型，通过添加回归层预测RNA扭转角。
使用正弦和余弦值预测角度以处理角度的周期性特性。
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 确保RNA-FM模块可以被正确导入
try:
    import fm
except ImportError:
    # 尝试添加fm模块路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    possible_paths = [
        os.path.join(project_root, "fm"),
        os.path.join(project_root, "RNA-FM", "fm"),
        os.path.join(os.path.dirname(project_root), "RNA-FM", "fm")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.append(os.path.dirname(path))
            try:
                import fm
                break
            except ImportError:
                continue
    else:
        raise ImportError(
            "无法导入fm模块。请确保RNA-FM已正确安装，或在环境变量中设置正确的路径。"
            "您可以尝试: pip install rna-fm 或手动指定RNA-FM的路径。"
        )

logger = logging.getLogger(__name__)

class RNATorsionPredictor(nn.Module):
    """
    基于RNA-FM的扭转角预测模型
    
    该模型使用预训练的RNA-FM提取RNA序列特征，然后通过回归层预测多种扭转角。
    为了处理角度的周期性特性，模型预测每种角度的正弦和余弦值，然后通过反正切函数恢复角度。
    
    特点:
    1. 使用RNA-FM提取RNA序列的上下文表示
    2. 冻结RNA-FM参数，只训练回归层
    3. 使用正弦/余弦预测处理角度周期性
    4. 为每种扭转角类型使用独立的回归头
    5. 支持同时预测多种扭转角
    """
    
    def __init__(self, 
                 rna_fm_model=None,
                 alphabet=None,
                 pretrained_model_path=None, 
                 torsion_types=None, 
                 hidden_dim=256, 
                 dropout=0.1,
                 layer_norm=True):
        """
        初始化扭转角预测模型
        
        参数:
            rna_fm_model: 预训练的RNA-FM模型对象（如果为None，则从path加载）
            alphabet: RNA-FM模型的字母表对象（如果为None，则从path加载）
            pretrained_model_path: RNA-FM预训练模型路径（当rna_fm_model为None时使用）
            torsion_types: 需要预测的扭转角类型列表，默认为标准RNA扭转角
            hidden_dim: 回归层隐藏层维度
            dropout: Dropout比例，用于防止过拟合
            layer_norm: 是否使用层归一化
        """
        super(RNATorsionPredictor, self).__init__()
        
        # 设置默认扭转角类型（如果未提供）
        if torsion_types is None:
            torsion_types = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi",  # 标准扭转角
            "eta", "theta", "eta'", "theta'",                            # 赝扭转角
            "v0", "v1", "v2", "v3", "v4"                                # 核糖构象角
            ]
        self.torsion_types = torsion_types
        
        # 加载RNA-FM模型（如果未提供）
        if rna_fm_model is None or alphabet is None:
            if pretrained_model_path is None:
                logger.info("未提供预训练模型路径，使用默认RNA-FM模型")
                self.rna_fm, self.alphabet = fm.pretrained.rna_fm_t12()
            else:
                logger.info(f"从路径加载RNA-FM模型: {pretrained_model_path}")
                self.rna_fm, self.alphabet = fm.pretrained.rna_fm_t12(model_location=pretrained_model_path)
        else:
            logger.info("使用提供的RNA-FM模型")
            self.rna_fm = rna_fm_model
            self.alphabet = alphabet
        
        # 冻结RNA-FM参数
        self._freeze_rnafm_parameters()
        
        # 获取RNA-FM输出维度
        self.embed_dim = 640  # RNA-FM的嵌入维度是640
        
        # 创建共享的特征提取层
        layers = []
        if layer_norm:
            layers.append(nn.LayerNorm(self.embed_dim))
        
        layers.extend([
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 为每种扭转角构建回归头
        self.regression_heads = nn.ModuleDict()
        for torsion_type in torsion_types:
            # 我们预测sin和cos，这样可以处理角度的周期性
            self.regression_heads[torsion_type] = nn.Linear(hidden_dim, 2)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重，使用适合回归任务的初始化方法"""
        # 特征提取器初始化
        for module in self.feature_extractor.modules():
            if isinstance(module, nn.Linear):
                # 使用He初始化（适用于GELU/ReLU激活函数）
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 回归头初始化
        for head in self.regression_heads.values():
            # 使用较小的初始值以获得更好的稳定性
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)
    
    def _freeze_rnafm_parameters(self):
        """冻结RNA-FM预训练模型的所有参数"""
        for param in self.rna_fm.parameters():
            param.requires_grad = False
        logger.info("RNA-FM参数已冻结")
    
    def forward(self, tokens):
        """
        前向传播
        
        参数:
            tokens: 输入的RNA序列token张量 [batch_size, seq_len]
        
        返回:
            predictions: 字典，键为扭转角类型，值为预测的角度
            sin_cos: 字典，键为扭转角类型，值为预测的sin和cos
        """
        # 使用RNA-FM提取特征
        with torch.no_grad():
            results = self.rna_fm(tokens, repr_layers=[12], need_head_weights=False)
        
        # 获取最后一层的表示
        embeddings = results["representations"][12]  # [batch_size, seq_len, embed_dim]
        
        # RNA-FM会添加特殊标记，我们需要去除它们
        # 通常第一个标记是<s>，最后一个标记是</s>
        # 去除特殊标记，只保留实际序列对应的表示
        embeddings = embeddings[:, 1:-1, :]  # [batch_size, seq_len-2, embed_dim]
        
        # 应用特征提取器
        features = self.feature_extractor(embeddings)  # [batch_size, seq_len-2, hidden_dim]
        
        # 预测各种扭转角
        predictions = {}
        sin_cos = {}
        
        for torsion_type in self.torsion_types:
            # 对每个残基位置进行预测
            output = self.regression_heads[torsion_type](features)  # [batch_size, seq_len-2, 2]
            
            # 分离sin和cos预测
            sin_pred = output[:, :, 0]  # [batch_size, seq_len-2]
            cos_pred = output[:, :, 1]  # [batch_size, seq_len-2]
            
            # 计算角度（弧度）
            angle_rad = torch.atan2(sin_pred, cos_pred)  # [batch_size, seq_len-2]
            
            # 转换为度
            angle_deg = angle_rad * 180.0 / torch.pi  # [batch_size, seq_len-2]
            
            # 存储结果
            predictions[torsion_type] = angle_deg
            sin_cos[torsion_type] = torch.stack([sin_pred, cos_pred], dim=-1)  # [batch_size, seq_len-2, 2]
        
        return predictions, sin_cos
    
    def save(self, path):
        """
        保存模型参数（不包括RNA-FM）
        
        参数:
            path: 保存路径
        """
        # 只保存训练过的部分（特征提取器和回归头）
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'regression_heads': self.regression_heads.state_dict(),
            'torsion_types': self.torsion_types
        }, path)
        logger.info(f"模型已保存到 {path}")
    
    def load(self, path):
        """
        加载模型参数
        
        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.regression_heads.load_state_dict(checkpoint['regression_heads'])
        logger.info(f"从 {path} 加载了模型")
        
        # 更新类中的扭转角类型（如果存在于checkpoint中）
        if 'torsion_types' in checkpoint:
            self.torsion_types = checkpoint['torsion_types']
            logger.info(f"加载了扭转角类型: {self.torsion_types}")
    
    def predict_single_sequence(self, sequence):
        """
        为单个RNA序列预测扭转角
        
        参数:
            sequence: RNA序列字符串
        
        返回:
            dict: 每种扭转角类型的预测角度
        """
        # 将序列转换为token
        batch_converter = self.alphabet.get_batch_converter()
        data = [("RNA", sequence)]
        _, _, tokens = batch_converter(data)
        
        # 移动到与模型相同的设备上
        device = next(self.parameters()).device
        tokens = tokens.to(device)
        
        # 进行预测
        self.eval()
        with torch.no_grad():
            predictions, _ = self.forward(tokens)
        
        # 提取每种扭转角的预测结果
        result = {}
        for angle_name, angle_preds in predictions.items():
            # 提取第一个（也是唯一的）序列的预测结果
            # 移除填充部分
            seq_len = min(len(sequence), angle_preds.shape[1])
            result[angle_name] = angle_preds[0, :seq_len].cpu().numpy()
        
        return result