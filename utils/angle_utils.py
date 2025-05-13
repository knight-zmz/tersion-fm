# utils/angle_utils.py
"""
角度计算和转换工具
"""

import numpy as np
import torch
import math

def degrees_to_radians(degrees):
    """将角度转换为弧度"""
    return degrees * math.pi / 180.0

def radians_to_degrees(radians):
    """将弧度转换为角度"""
    return radians * 180.0 / math.pi

def normalize_angle_degrees(degrees):
    """将角度归一化到[-180, 180]范围"""
    return ((degrees + 180.0) % 360.0) - 180.0

def angle_difference_degrees(angle1, angle2):
    """
    计算两个角度之间的最小差异（度）
    
    Args:
        angle1, angle2: 角度（度），可以是标量或数组
    
    Returns:
        diff: 角度差异，范围[-180, 180]
    """
    # 转换为NumPy数组以确保一致性
    angle1 = np.asarray(angle1)
    angle2 = np.asarray(angle2)
    
    # 计算差异并取模
    diff = (angle1 - angle2) % 360.0
    
    # 使用where进行元素级比较和替换
    # 如果diff > 180.0，则diff -= 360.0
    return np.where(diff > 180.0, diff - 360.0, diff)

def compute_circular_correlation(pred_angles, true_angles, mask=None):
    """
    计算预测角度和真实角度之间的圆形相关系数
    
    Args:
        pred_angles: 预测角度（度）
        true_angles: 真实角度（度）
        mask: 可选的掩码，指示有效值
    
    Returns:
        corr: 圆形相关系数
    """
    # 转换为NumPy数组，如果是PyTorch张量
    if isinstance(pred_angles, torch.Tensor):
        pred_angles = pred_angles.detach().cpu().numpy()
    if isinstance(true_angles, torch.Tensor):
        true_angles = true_angles.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 转换为弧度
    pred_rad = np.radians(pred_angles)
    true_rad = np.radians(true_angles)
    
    # 计算sin和cos
    pred_sin = np.sin(pred_rad)
    pred_cos = np.cos(pred_rad)
    true_sin = np.sin(true_rad)
    true_cos = np.cos(true_rad)
    
    # 应用掩码，如果提供
    if mask is not None:
        # 确保形状兼容
        try:
            pred_sin = pred_sin * mask
            pred_cos = pred_cos * mask
            true_sin = true_sin * mask
            true_cos = true_cos * mask
        except ValueError:
            # 形状不兼容时，尝试广播
            mask_broadcast = np.broadcast_to(mask, pred_sin.shape)
            pred_sin = pred_sin * mask_broadcast
            pred_cos = pred_cos * mask_broadcast
            true_sin = true_sin * mask_broadcast
            true_cos = true_cos * mask_broadcast
        
        # 计算有效样本数
        n_valid = np.sum(mask)
        if n_valid == 0:
            return 0.0
    else:
        # 无掩码时使用全部元素
        n_valid = np.size(pred_sin)
    
    # 计算相关系数
    sin_corr = np.sum(pred_sin * true_sin) / n_valid
    cos_corr = np.sum(pred_cos * true_cos) / n_valid
    
    # 合并相关系数
    return (sin_corr + cos_corr) / 2.0

def compute_mae_degrees(pred_angles, true_angles, mask=None):
    """
    计算角度的平均绝对误差（度）
    
    Args:
        pred_angles: 预测角度（度）
        true_angles: 真实角度（度）
        mask: 可选的掩码，指示有效值
    
    Returns:
        mae: 平均绝对误差（度）
    """
    # 转换为NumPy数组，如果是PyTorch张量
    if isinstance(pred_angles, torch.Tensor):
        pred_angles = pred_angles.detach().cpu().numpy()
    if isinstance(true_angles, torch.Tensor):
        true_angles = true_angles.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 计算角度差异
    diff = np.abs(angle_difference_degrees(pred_angles, true_angles))
    
    # 应用掩码，如果提供
    if mask is not None:
        # 确保掩码和diff形状匹配
        if mask.shape != diff.shape:
            # 如果维度不匹配，尝试广播
            try:
                masked_diff = diff * mask
            except ValueError:
                # 如果广播失败，确保维度匹配再进行操作
                mask = np.broadcast_to(mask, diff.shape)
                masked_diff = diff * mask
        else:
            masked_diff = diff * mask
            
        # 计算平均误差（只考虑有效值）
        mask_sum = np.sum(mask)
        if mask_sum > 0:
            mae = np.sum(masked_diff) / mask_sum
        else:
            mae = 0.0
    else:
        # 无掩码时的平均误差
        mae = np.mean(diff)
    
    return mae