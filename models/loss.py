# models/loss.py
"""
角度预测专用损失函数，处理角度的周期性特性
"""

import torch
import torch.nn as nn
import math

class AngularLoss(nn.Module):
    """
    角度预测的周期性损失函数
    
    通过比较正弦和余弦值而不是直接比较角度值，
    有效地处理角度的周期性（例如，359度和1度的实际差异很小）
    """
    def __init__(self, weight=1.0):
        """
        初始化角度损失
        
        参数:
            weight: 损失权重
        """
        super(AngularLoss, self).__init__()
        self.weight = weight
    
    def forward(self, sin_cos_pred, angle_target, mask=None):
        """
        计算角度损失
        
        参数:
            sin_cos_pred: 预测的sin和cos值 [batch_size, seq_len, 2]
            angle_target: 目标角度（度） [batch_size, seq_len]
            mask: 掩码张量，1表示有效值，0表示缺失值 [batch_size, seq_len]
        
        返回:
            loss: 损失值
        """
        # 将目标角度转换为弧度
        angle_rad = angle_target * math.pi / 180.0
        
        # 计算目标角度的sin和cos
        sin_target = torch.sin(angle_rad)
        cos_target = torch.cos(angle_rad)
        
        # 提取预测的sin和cos
        sin_pred = sin_cos_pred[:, :, 0]
        cos_pred = sin_cos_pred[:, :, 1]
        
        # 计算sin和cos的均方误差
        sin_loss = (sin_pred - sin_target) ** 2
        cos_loss = (cos_pred - cos_target) ** 2
        
        # 组合损失
        combined_loss = sin_loss + cos_loss
        
        # 应用掩码（如果提供）
        if mask is not None:
            combined_loss = combined_loss * mask
            # 计算平均损失（只考虑有效值）
            total_valid = mask.sum()
            if total_valid > 0:
                return self.weight * combined_loss.sum() / total_valid
            else:
                return torch.tensor(0.0, device=combined_loss.device)
        else:
            # 无掩码时的平均损失
            return self.weight * combined_loss.mean()

class TotalAngularLoss(nn.Module):
    """
    所有扭转角的总损失
    
    整合多种扭转角的损失，可为不同角度类型设置不同的权重
    """
    def __init__(self, torsion_types, weights=None):
        """
        初始化
        
        参数:
            torsion_types: 扭转角类型列表
            weights: 可选的每种角度的权重字典
        """
        super(TotalAngularLoss, self).__init__()
        self.torsion_types = torsion_types
        
        # 默认每种角度的权重为1
        if weights is None:
            self.weights = {angle: 1.0 for angle in torsion_types}
        else:
            self.weights = weights
        
        # 为每种角度创建损失函数
        self.angle_losses = nn.ModuleDict()
        for angle_name in torsion_types:
            self.angle_losses[angle_name] = AngularLoss(weight=self.weights.get(angle_name, 1.0))
    
    def forward(self, sin_cos_preds, angle_targets, masks):
        """
        计算总损失
        
        参数:
            sin_cos_preds: 字典，键为角度名，值为预测的sin和cos
            angle_targets: 字典，键为角度名，值为目标角度
            masks: 字典，键为角度名，值为掩码
        
        返回:
            total_loss: 总损失
            loss_dict: 每种角度的损失字典
        """
        total_loss = 0.0
        loss_dict = {}
        
        for angle_name in self.torsion_types:
            if angle_name in sin_cos_preds and angle_name in angle_targets:
                loss = self.angle_losses[angle_name](
                    sin_cos_preds[angle_name],
                    angle_targets[angle_name],
                    masks.get(angle_name, None)
                )
                
                total_loss += loss
                loss_dict[angle_name] = loss.item()
        
        return total_loss, loss_dict