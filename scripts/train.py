# scripts/train.py
"""
模型训练脚本
"""

import os
import sys
import logging
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import argparse
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('D:\\source\\myvscode\\python_work\\RNA-FM')
from config.config import Config
from data.dataset import RNATorsionDataset, create_data_loaders
from models.torsion_predictor import RNATorsionPredictor
from models.loss import TotalAngularLoss
from utils.evaluation import evaluate_model
import fm

def setup_logger(log_dir):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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

def seed_everything(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(cfg):
    """
    训练模型
    
    Args:
        cfg: 配置对象
    """
    # 设置随机种子
    seed_everything(42)
    
    # 设置设备
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载RNA-FM模型
    logging.info("加载RNA-FM模型...")
    rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
    rna_fm_model.eval()  # 设为评估模式
    rna_fm_model.to(device)
    logging.info("RNA-FM模型加载完成")
    
    # 创建数据集
    logging.info(f"创建数据集，从目录: {cfg.DATA_DIR}")
    dataset = RNATorsionDataset(
        cfg.DATA_DIR, 
        alphabet, 
        cfg.TORSION_TYPES,
        cache_dir=os.path.join(cfg.OUTPUT_DIR, "cache")
    )
    
    # 创建数据加载器
    logging.info("创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
        num_workers=cfg.NUM_WORKERS
    )
    
    # 创建模型
    logging.info("创建扭转角预测模型...")
    model = RNATorsionPredictor(
        rna_fm_model,
        alphabet,
        torsion_types=cfg.TORSION_TYPES,
        hidden_dim=cfg.HIDDEN_DIM,
        dropout=cfg.DROPOUT
    )
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = TotalAngularLoss(cfg.TORSION_TYPES)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    # 创建TensorBoard写入器
    tb_dir = os.path.join(cfg.EXPERIMENT_DIR, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    
    # 保存检查点的目录
    checkpoint_dir = os.path.join(cfg.EXPERIMENT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 记录最佳验证性能
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 10
    
    # 训练循环
    logging.info(f"开始训练，共{cfg.NUM_EPOCHS}个epoch")
    for epoch in range(cfg.NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_loss_dict = {angle: 0.0 for angle in cfg.TORSION_TYPES}
        
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移到设备上
            tokens = batch['tokens'].to(device)
            
            # 前向传播
            predictions, sin_cos_preds = model(tokens)
            
            # 计算损失
            angle_targets = {angle: batch['angles'][angle].to(device) for angle in cfg.TORSION_TYPES if angle in batch['angles']}
            angle_masks = {angle: batch['masks'][angle].to(device) for angle in cfg.TORSION_TYPES if angle in batch['masks']}
            
            loss, loss_dict = criterion(sin_cos_preds, angle_targets, angle_masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            for angle, angle_loss in loss_dict.items():
                train_loss_dict[angle] += angle_loss
            
            # 记录进度
            if (batch_idx + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_loss_dict = {angle: loss / len(train_loader) for angle, loss in train_loss_dict.items()}
        
        # 记录训练损失到TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        for angle, loss in train_loss_dict.items():
            writer.add_scalar(f"Loss_train/{angle}", loss, epoch)
        
        end_time = time.time()
        logging.info(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} 训练完成，耗时: {end_time - start_time:.2f}秒, 平均损失: {train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_loss_dict = {angle: 0.0 for angle in cfg.TORSION_TYPES}
        
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                
                # 前向传播
                predictions, sin_cos_preds = model(tokens)
                
                # 计算损失
                angle_targets = {angle: batch['angles'][angle].to(device) for angle in cfg.TORSION_TYPES if angle in batch['angles']}
                angle_masks = {angle: batch['masks'][angle].to(device) for angle in cfg.TORSION_TYPES if angle in batch['masks']}
                
                loss, loss_dict = criterion(sin_cos_preds, angle_targets, angle_masks)
                
                # 累计损失
                val_loss += loss.item()
                for angle, angle_loss in loss_dict.items():
                    val_loss_dict[angle] += angle_loss
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        val_loss_dict = {angle: loss / len(val_loader) for angle, loss in val_loss_dict.items()}
        
        # 记录验证损失到TensorBoard
        writer.add_scalar("Loss/val", val_loss, epoch)
        for angle, loss in val_loss_dict.items():
            writer.add_scalar(f"Loss_val/{angle}", loss, epoch)
        
        logging.info(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} 验证完成，平均损失: {val_loss:.4f}")
        
        # 检查是否需要保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            model.save(best_model_path)
            logging.info(f"最佳模型已保存，验证损失: {val_loss:.4f}")
        else:
            early_stop_counter += 1
            logging.info(f"验证损失未改善，早停计数器: {early_stop_counter}/{early_stop_patience}")
        
        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth")
            model.save(checkpoint_path)
            logging.info(f"Epoch {epoch+1} 检查点已保存")
        
        # 早停
        if early_stop_counter >= early_stop_patience:
            logging.info(f"早停触发，{early_stop_patience}个epoch未改善")
            break
    
    logging.info("训练完成")
    
    # 在测试集上评估最佳模型
    logging.info("加载最佳模型并在测试集上评估...")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    model.load(best_model_path)
    
    print("\n开始评估模型...")
    try:
        # 在最佳模型评估部分，添加测试结果目录的定义
        test_results_dir = os.path.join(cfg.EXPERIMENT_DIR, "test_results")
        os.makedirs(test_results_dir, exist_ok=True)
        logging.info(f"测试结果将保存到: {test_results_dir}")
        metrics = evaluate_model(model, test_loader, device, cfg.TORSION_TYPES, test_results_dir)
        
        print(f"评估完成，详细指标: {metrics}")
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        metrics = {}

    logging.info("评估完成，指标:")
    for name, value in metrics.items():
        logging.info(f"{name}: {value:.4f}")
    
    # 关闭TensorBoard写入器
    writer.close()

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练RNA扭转角预测模型")
    parser.add_argument("--data_dir", type=str, help="包含pkl文件的数据目录")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda", help="设备（'cuda'或'cpu'）")
    
    args = parser.parse_args()
    
    # 创建配置对象
    cfg = Config()
    
    # 更新配置（如果提供了命令行参数）
    if args.data_dir:
        cfg.DATA_DIR = args.data_dir
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size
    if args.num_epochs:
        cfg.NUM_EPOCHS = args.num_epochs
    if args.learning_rate:
        cfg.LEARNING_RATE = args.learning_rate
    if args.device:
        cfg.DEVICE = args.device
    
    # 设置日志记录器
    logger = setup_logger(os.path.join(cfg.EXPERIMENT_DIR, "logs"))
    
    # 记录配置
    logging.info(f"配置: {vars(cfg)}")
    
    # 训练模型
    train_model(cfg)

if __name__ == "__main__":
    main()