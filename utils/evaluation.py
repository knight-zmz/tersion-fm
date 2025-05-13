# utils/evaluation.py
"""
模型评估工具
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from .angle_utils import compute_circular_correlation, compute_mae_degrees

logger = logging.getLogger(__name__)


def evaluate_model(model, data_loader, device, torsion_types, output_dir=None):
    model.eval()
    
    # 初始化结果字典
    all_predictions = {angle: [] for angle in torsion_types}
    all_targets = {angle: [] for angle in torsion_types}
    all_masks = {angle: [] for angle in torsion_types}
    all_seq_lens = {angle: [] for angle in torsion_types}
    pdb_ids = []
    chain_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            tokens = batch['tokens'].to(device)
            pdb_ids.extend(batch['pdb_ids'])
            chain_ids.extend(batch['chain_ids'])
            
            # 获取预测
            predictions, _ = model(tokens)

            # 处理每种角度类型
            for angle_name in torsion_types:
                if angle_name in batch['angles']:
                    angle_target = batch['angles'][angle_name].to(device)
                    angle_mask = batch['masks'][angle_name].to(device)
                    pred = predictions[angle_name]
                    
                    # 分别处理批次中的每个序列
                    batch_size = angle_target.shape[0]
                    for i in range(batch_size):
                        target_len = angle_target[i].shape[0]
                        pred_len = pred[i].shape[0]
                        min_len = min(target_len, pred_len)
                        
                        # 存储单个序列的预测和目标
                        all_predictions[angle_name].append(pred[i, :min_len].unsqueeze(0))
                        all_targets[angle_name].append(angle_target[i, :min_len].unsqueeze(0))
                        all_masks[angle_name].append(angle_mask[i, :min_len].unsqueeze(0))
                        all_seq_lens[angle_name].append(min_len)
    
    # 计算指标
    metrics = {}
    
    # 按角度类型分组
    angle_groups = {
        "backbone": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
        "chi": ["chi"],
        "pseudo": ["eta", "theta", "eta'", "theta'"],
        "ribose": ["v0", "v1", "v2", "v3", "v4"]
    }
    
    # 创建角度类型到组的映射，便于后续查找
    angle_to_group = {}
    for group_name, angles in angle_groups.items():
        for angle in angles:
            angle_to_group[angle] = group_name
    
    # 定义表格格式和分隔符
    header = f"{'角度类型':<10} {'MAE (°)':<12} {'循环相关':<12}"
    separator = "-" * 36
    
    # 打印标题
    print("\n" + "=" * 40)
    print(f"{'模型评估结果':^40}")
    print("=" * 40)
    
    # 记录主要评估信息到日志，但不打印表格
    logger.info("开始计算各角度评估指标")
    
    # 打印各角度详细指标
    print(f"\n{'各角度详细指标':^40}")
    print(separator)
    print(header)
    print(separator)
    
    # 计算每个角度的指标并排序显示
    for angle_name in sorted(torsion_types, key=lambda x: (angle_to_group.get(x, ''), x)):
        if all_predictions[angle_name]:
            # 分别计算每个序列的指标并平均
            mae_values = []
            corr_values = []
            
            # 计算每个序列的指标
            for i in range(len(all_predictions[angle_name])):
                pred = all_predictions[angle_name][i].squeeze(0)  # 移除批次维度
                target = all_targets[angle_name][i].squeeze(0)    # 移除批次维度
                mask = all_masks[angle_name][i].squeeze(0)        # 移除批次维度
                
                # 确保是NumPy数组
                pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
                target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
                mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                
                # 计算指标
                try:
                    mae = compute_mae_degrees(pred_np, target_np, mask_np)
                    corr = compute_circular_correlation(pred_np, target_np, mask_np)
                    
                    if not np.isnan(mae) and not np.isinf(mae) and mae is not None:
                        mae_values.append(mae)
                    if not np.isnan(corr) and not np.isinf(corr) and corr is not None:
                        corr_values.append(corr)
                except Exception as e:
                    logger.warning(f"计算{angle_name}指标时发生错误: {str(e)}")

            if mae_values:
                avg_mae = np.mean(mae_values)
                metrics[f"{angle_name}_mae"] = avg_mae
                
                if corr_values:
                    avg_corr = np.mean(corr_values)
                    metrics[f"{angle_name}_corr"] = avg_corr
                    
                    # 只在控制台输出表格格式
                    row = f"{angle_name:<10} {avg_mae:<12.2f} {avg_corr:<12.4f}"
                    print(row)
                    
                    # 在日志中只记录关键信息，不记录表格
                    logger.info(f"{angle_name} MAE: {avg_mae:.4f}, 相关性: {avg_corr:.4f}")
                else:
                    row = f"{angle_name:<10} {avg_mae:<12.2f} {'N/A':<12}"
                    print(row)
                    logger.info(f"{angle_name} MAE: {avg_mae:.4f}, 相关性: N/A")
            else:
                row = f"{angle_name:<10} {'N/A':<12} {'N/A':<12}"
                print(row)
                logger.info(f"{angle_name}: 无有效数据")
    
    print(separator)
    
    # 计算组平均指标
    group_metrics = {}
    
    print(f"\n{'按组平均指标':^40}")
    print(separator)
    print(f"{'组名称':<10} {'平均MAE (°)':<12} {'平均相关':<12}")
    print(separator)
    
    logger.info("==== 按组评估结果 ====")
    
    for group_name, angles in angle_groups.items():
        group_mae = []
        group_corr = []
        for angle in angles:
            if f"{angle}_mae" in metrics:
                group_mae.append(metrics[f"{angle}_mae"])
            if f"{angle}_corr" in metrics:
                group_corr.append(metrics[f"{angle}_corr"])
        
        if group_mae:
            avg_mae = np.mean(group_mae)
            group_metrics[f"{group_name}_avg_mae"] = avg_mae
            
            if group_corr:
                avg_corr = np.mean(group_corr)
                group_metrics[f"{group_name}_avg_corr"] = avg_corr
                
                row = f"{group_name:<10} {avg_mae:<12.2f} {avg_corr:<12.4f}"
                print(row)
            else:
                row = f"{group_name:<10} {avg_mae:<12.2f} {'N/A':<12}"
                print(row)
                
            # 只在日志中记录一次
            logger.info(f"{group_name} 平均MAE: {avg_mae:.2f}°, 平均相关性: {group_metrics.get(f'{group_name}_avg_corr', 'N/A')}")
    
    # 合并组指标到总结果
    metrics.update(group_metrics)
    print(separator)
    
    # 计算总体平均MAE
    angle_metrics = [(angle, metrics.get(f"{angle}_mae", np.nan)) for angle in torsion_types]
    valid_metrics = [(angle, value) for angle, value in angle_metrics if not np.isnan(value)]
    
    print(f"\n{'总体指标':^40}")
    print(separator)
    
    if valid_metrics:
        avg_mae = np.mean([value for _, value in valid_metrics])
        metrics["avg_mae"] = avg_mae
        print(f"{'总体平均MAE (°)':<20} {avg_mae:<12.2f}")
        logger.info(f"总体平均 MAE: {avg_mae:.2f}°")
    else:
        print("无法计算平均MAE (没有有效指标)")
        logger.info("无法计算平均MAE (没有有效指标)")
    
    print(separator)
    print("\n评估完成！")
    
    # 如果提供了输出目录则保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存指标
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(output_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n指标已保存到: {metrics_path}")
        
        # 可视化 - 解决中文字体问题
        try:
            # 配置中文字体支持
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']  # 优先使用中文黑体
            plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
            
            # 绘制各角度MAE柱状图
            plt.figure(figsize=(12, 6))
            angles = [angle for angle in torsion_types if f"{angle}_mae" in metrics]
            maes = [metrics[f"{angle}_mae"] for angle in angles]
            
            # 按组为柱状图上色
            colors = {'backbone': 'royalblue', 'chi': 'green', 'pseudo': 'orange', 'ribose': 'crimson'}
            bar_colors = [colors[angle_to_group.get(angle, 'other')] for angle in angles]
            
            plt.bar(angles, maes, color=bar_colors)
            
            # 使用英文标题避免字体问题
            plt.title('Mean Absolute Error (MAE) by Angle Type')
            plt.ylabel('MAE (degrees)')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[group], label=group) for group in angle_groups.keys()]
            plt.legend(handles=legend_elements)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "angle_mae.png"))
            print(f"生成可视化图表: {os.path.join(output_dir, 'angle_mae.png')}")
            
            # 绘制循环相关系数柱状图
            plt.figure(figsize=(12, 6))
            angles_corr = [angle for angle in torsion_types if f"{angle}_corr" in metrics]
            corrs = [metrics[f"{angle}_corr"] for angle in angles_corr]
            
            bar_colors = [colors[angle_to_group.get(angle, 'other')] for angle in angles_corr]
            
            plt.bar(angles_corr, corrs, color=bar_colors)
            plt.title('Circular Correlation by Angle Type')
            plt.ylabel('Correlation Coefficient')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(handles=legend_elements)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "angle_correlation.png"))
            print(f"生成可视化图表: {os.path.join(output_dir, 'angle_correlation.png')}")
            
        except Exception as e:
            logger.warning(f"生成可视化图表时出错: {str(e)}")
            print(f"生成可视化图表时出错: {str(e)}")
    
    # 简化最终输出的打印，避免打印完整的metrics字典（这会造成混乱）
    logger.info(f"评估完成，总计指标项: {len(metrics.keys())}项")
    
    return metrics