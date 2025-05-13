# scripts/predict.py
"""
模型预测脚本
"""

import os
import sys
import logging
import torch
import argparse
import json
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.preprocessing import process_pdb_file
from models.torsion_predictor import RNATorsionPredictor
import fm

def setup_logger(log_dir):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 创建文件处理程序
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 添加处理程序到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def predict(input_file, model_path, output_dir, device="cuda"):
    """
    预测RNA扭转角
    
    Args:
        input_file: 输入的pkl文件路径
        model_path: 模型检查点路径
        output_dir: 输出目录
        device: 设备（'cuda'或'cpu'）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载RNA-FM模型
    logging.info("加载RNA-FM模型...")
    rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
    rna_fm_model.eval()  # 设为评估模式
    rna_fm_model.to(device)
    logging.info("RNA-FM模型加载完成")
    
    # 加载检查点以获取扭转角类型
    logging.info(f"加载模型检查点: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    torsion_types = checkpoint['torsion_types']
    
    # 创建模型
    model = RNATorsionPredictor(rna_fm_model, alphabet, torsion_types)
    
    # 加载模型参数
    model.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    model.regression_heads.load_state_dict(checkpoint['regression_heads'])
    model.to(device)
    model.eval()
    
    # 处理输入文件
    logging.info(f"处理输入文件: {input_file}")
    result = process_pdb_file(input_file)
    
    if result is None:
        logging.error(f"无法处理文件: {input_file}")
        return
    
    # 提取序列
    sequence = result['sequence']
    logging.info(f"序列长度: {len(sequence)}")
    
    # 将序列转换为token
    data = [("RNA", sequence)]
    _, _, tokens = alphabet.get_batch_converter()(data)
    tokens = tokens.to(device)
    
    # 预测
    logging.info("进行预测...")
    with torch.no_grad():
        predictions, _ = model(tokens)
    
    # 准备结果
    results = []
    
    for i in range(len(sequence)):
        residue_result = {
            'residue_id': result['sorted_residue_ids'][i] if i < len(result['sorted_residue_ids']) else i + 1,
            'residue': sequence[i] if i < len(sequence) else "X",
        }
        
        # 添加每种扭转角的预测值
        for angle_name in torsion_types:
            if angle_name in predictions and i < predictions[angle_name].shape[1]:
                residue_result[f"{angle_name}_pred"] = float(predictions[angle_name][0, i].cpu().numpy())
            else:
                residue_result[f"{angle_name}_pred"] = None
        
        # 如果有真实值，也添加
        if 'torsion_angles' in result:
            for angle_name in torsion_types:
                if angle_name in result['torsion_angles'] and i < len(result['torsion_angles'][angle_name]):
                    # 检查掩码是否为有效值
                    if result['torsion_masks'][angle_name][i] > 0:
                        residue_result[f"{angle_name}_true"] = result['torsion_angles'][angle_name][i]
                    else:
                        residue_result[f"{angle_name}_true"] = None
                else:
                    residue_result[f"{angle_name}_true"] = None
        
        results.append(residue_result)
    
    # 保存结果
    pdb_id = os.path.basename(input_file).split('.')[0]
    
    # 保存为CSV
    csv_path = os.path.join(output_dir, f"{pdb_id}_predictions.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    logging.info(f"预测结果已保存到: {csv_path}")
    
    # 保存为JSON
    json_path = os.path.join(output_dir, f"{pdb_id}_predictions.json")
    with open(json_path, 'w') as f:
        json.dump({
            'pdb_id': pdb_id,
            'sequence': sequence,
            'predictions': results
        }, f, indent=2)
    logging.info(f"预测结果已保存到: {json_path}")
    
    logging.info("预测完成")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="预测RNA扭转角")
    parser.add_argument("--input_file", type=str, required=True, help="输入的pkl文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备（'cuda'或'cpu'）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(os.path.join(args.output_dir, "logs"))
    
    # 记录参数
    logging.info(f"输入文件: {args.input_file}")
    logging.info(f"模型路径: {args.model_path}")
    logging.info(f"输出目录: {args.output_dir}")
    logging.info(f"设备: {args.device}")
    
    # 执行预测
    predict(args.input_file, args.model_path, args.output_dir, args.device)

if __name__ == "__main__":
    main()