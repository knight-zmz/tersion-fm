# main.py
"""
RNA-FM-Torsion主程序
"""

import os
import argparse
import logging
from datetime import datetime
import torch

from config.config import Config
from scripts.train import train_model
from scripts.predict import predict

def setup_logger(log_dir):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RNA-FM-Torsion: 基于RNA-FM的RNA扭转角预测")
    
    # 添加帮助信息
    parser.add_argument('--help-examples', action='store_true', 
                      help='显示命令示例')
    
    subparsers = parser.add_subparsers(dest="command", help="子命令", required=True)
    
    # 训练子命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data_dir", type=str, required=True, help="包含pkl文件的数据目录")
    train_parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    train_parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    train_parser.add_argument("--num_epochs", type=int, default=20, help="训练轮数")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    train_parser.add_argument("--device", type=str, default="cuda", help="设备（'cuda'或'cpu'）")
    
    # 预测子命令
    predict_parser = subparsers.add_parser("predict", help="预测扭转角")
    predict_parser.add_argument("--input_file", type=str, required=True, help="输入的pkl文件路径")
    predict_parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    predict_parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    predict_parser.add_argument("--device", type=str, default="cuda", help="设备（'cuda'或'cpu'）")
    
    try:
        args = parser.parse_args()
        
        # 显示使用示例
        if hasattr(args, 'help_examples') and args.help_examples:
            print("\n使用示例:")
            print("  训练模型:")
            print("    python main.py train --data_dir ./data/pkl_files --output_dir ./output")
            print("\n  预测扭转角:")
            print("    python main.py predict --input_file ./data/example.pkl --model_path ./output/best_model.pth --output_dir ./predictions\n")
            return
        
        # 创建配置对象
        cfg = Config()
        
        # 设置日志记录器
        if args.command == "train" and hasattr(args, 'output_dir') and args.output_dir:
            cfg.OUTPUT_DIR = args.output_dir
        elif args.command == "predict" and hasattr(args, 'output_dir') and args.output_dir:
            cfg.OUTPUT_DIR = args.output_dir
        
        logger = setup_logger(os.path.join(cfg.OUTPUT_DIR, "logs"))
        
        # 根据命令执行相应的操作
        if args.command == "train":
            # 更新配置
            if hasattr(args, 'data_dir') and args.data_dir:
                cfg.DATA_DIR = args.data_dir
            if hasattr(args, 'batch_size'):
                cfg.BATCH_SIZE = args.batch_size
            if hasattr(args, 'num_epochs'):
                cfg.NUM_EPOCHS = args.num_epochs
            if hasattr(args, 'learning_rate'):
                cfg.LEARNING_RATE = args.learning_rate
            if hasattr(args, 'device'):
                cfg.DEVICE = args.device
            
            # 记录配置
            logging.info(f"配置: {vars(cfg)}")
            
            # 训练模型
            train_model(cfg)
        
        elif args.command == "predict":
            # 执行预测
            if (hasattr(args, 'input_file') and args.input_file and 
                hasattr(args, 'model_path') and args.model_path and 
                hasattr(args, 'output_dir') and args.output_dir):
                predict(args.input_file, args.model_path, args.output_dir, 
                       args.device if hasattr(args, 'device') else "cuda")
            else:
                logging.error("预测需要提供 --input_file, --model_path 和 --output_dir 参数")
                parser.print_help()
    
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("\n正确的使用方法:")
        print("  训练模型:")
        print("    python main.py train --data_dir ./data/pkl_files --output_dir ./output")
        print("\n  预测扭转角:")
        print("    python main.py predict --input_file ./data/example.pkl --model_path ./output/best_model.pth --output_dir ./predictions\n")
        
        # 在较新的Python版本(3.9+)中，required=True属性可能会引起问题
        # 提供更详细的错误信息
        if "required" in str(e):
            print("注意: 必须指定子命令 'train' 或 'predict'")
        
        raise

if __name__ == "__main__":
    main()