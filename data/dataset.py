# data/dataset.py
"""
数据集定义模块
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import glob
import logging
import numpy as np
from .preprocessing import process_pdb_file

logger = logging.getLogger(__name__)

class RNATorsionDataset(Dataset):
    """RNA扭转角数据集"""
    
    def __init__(self, data_dir, alphabet, torsion_types, cache_dir=None):
        """
        初始化数据集
        
        Args:
            data_dir: 包含pkl文件的目录
            alphabet: RNA-FM的字母表
            torsion_types: 需要预测的扭转角类型列表
            cache_dir: 缓存目录，如果提供则缓存处理后的数据
        """
        self.data_dir = data_dir
        self.alphabet = alphabet
        self.torsion_types = torsion_types
        self.cache_dir = cache_dir
        self.batch_converter = alphabet.get_batch_converter()
        
        # 获取所有pkl文件路径
        self.pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        logger.info(f"找到{len(self.pkl_files)}个pkl文件")
        
        # 预处理数据
        self.data = []
        self._load_and_process_data()
        
        logger.info(f"成功加载{len(self.data)}个RNA样本")
    
    def _load_and_process_data(self):
        """加载并预处理所有pkl文件"""
        # 如果存在缓存，则从缓存加载
        cache_path = os.path.join(self.cache_dir, "processed_data.pt")
        if self.cache_dir and os.path.exists(cache_path):
            logger.info(f"从缓存加载数据: {cache_path}")
            try:
                # 使用PyTorch 2.6+的新weights_only=False标志
                import torch.serialization
                with torch.serialization.safe_globals([np.core.multiarray.scalar, np.float64, np.float32]):
                    self.data = torch.load(cache_path)
            except Exception as e:
                logger.warning(f"加载缓存失败: {str(e)}")
                logger.warning("删除旧缓存并重新处理数据...")
                os.remove(cache_path)
                self.data = []
            
            if len(self.data) == 0:
                logger.warning("缓存的数据集为空或加载失败!")
            else:
                logger.info(f"从缓存加载了 {len(self.data)} 个样本")
                return
        
        # 查找所有pkl文件
        self.pkl_files = glob.glob(os.path.join(self.data_dir, "*.pkl")) + \
                        glob.glob(os.path.join(self.data_dir, "*.pt"))
        logger.info(f"找到 {len(self.pkl_files)} 个数据文件")
        
        if len(self.pkl_files) == 0:
            raise FileNotFoundError(f"在目录 {self.data_dir} 中未找到数据文件")
        
        # 处理所有文件
        for file_path in self.pkl_files:
            try:
                # 尝试加载文件
                if file_path.endswith('.pt'):
                    # 直接加载PyTorch保存的数据
                    processed_data = torch.load(file_path)
                    if isinstance(processed_data, list):
                        self.data.extend(processed_data)
                        logger.info(f"从 {file_path} 加载了 {len(processed_data)} 个结构")
                        continue
                
                # 处理PKL文件
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # 检查数据类型
                if isinstance(data, dict):
                    file_name = os.path.basename(file_path)
                    
                    # 检查是否为Training_Dict_single格式
                    if "Training_Dict_single" in file_name or any(k in ['2', '3', '4', '5'] for k in data.keys()):
                        logger.info(f"检测到训练字典格式: {file_name}")
                        from .adapters import adapt_training_dict_single
                        adapted_data = adapt_training_dict_single(data)
                        if adapted_data:
                            self.data.extend(adapted_data)
                            logger.info(f"适配处理成功: {len(adapted_data)} 个样本")
                            continue
                    
                    # 尝试标准处理
                    result = process_pdb_file(file_path)
                    if result is not None:
                        self.data.append(result)
                
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {str(e)}")
        
        logger.info(f"数据加载完成: 总计 {len(self.data)} 个样本")
        
        if len(self.data) == 0:
            raise ValueError(f"未能加载任何有效数据，请检查数据格式")
        
       # 保存到缓存时，也更新保存操作
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir, "processed_data.pt")
            try:
                 # 在此处重新导入torch确保可用
                import torch
        
                # 在保存前将NumPy标量转换为Python原生类型
                processed_data = []
                for item in self.data:
                    # 深拷贝以避免修改原始数据
                    import copy
                    item_copy = copy.deepcopy(item)
                    
                    # 转换torsion_angles中的NumPy标量
                    for angle_type in item_copy.get('torsion_angles', {}):
                        item_copy['torsion_angles'][angle_type] = [
                            float(val) if isinstance(val, np.number) else val 
                            for val in item_copy['torsion_angles'][angle_type]
                        ]
                        
                    # 转换torsion_masks中的NumPy标量
                    for angle_type in item_copy.get('torsion_masks', {}):
                        item_copy['torsion_masks'][angle_type] = [
                            float(val) if isinstance(val, np.number) else val 
                            for val in item_copy['torsion_masks'][angle_type]
                        ]
                        
                    processed_data.append(item_copy)
                    
                torch.save(processed_data, cache_path)
                logger.info(f"数据已缓存到: {cache_path}")
            except Exception as e:
                logger.error(f"保存缓存失败: {str(e)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """返回一个样本"""
        sample = self.data[idx]
        
        # 提取序列
        seq = sample['sequence']
        
        # 将序列转换为token
        data = [("RNA", seq)]
        _, _, tokens = self.batch_converter(data)
        
        # 提取扭转角和掩码
        angles = {}
        masks = {}
        
        for angle_name in self.torsion_types:
            if angle_name in sample['torsion_angles']:
                angles[angle_name] = torch.tensor(sample['torsion_angles'][angle_name], dtype=torch.float)
                masks[angle_name] = torch.tensor(sample['torsion_masks'][angle_name], dtype=torch.float)
        
        return {
            'pdb_id': sample['pdb_id'],
            'chain_id': sample['chain_id'],
            'sequence': seq,
            'tokens': tokens[0],
            'angles': angles,
            'masks': masks
        }

def create_data_loaders(dataset, batch_size, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, num_workers=4):
    """
    创建训练、验证和测试数据加载器
    """
    # 检查比例之和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例之和必须为1"
    
    # 检查数据集大小
    dataset_size = len(dataset)
    logger.info(f"数据集大小: {dataset_size}")
    
    if dataset_size == 0:
        raise ValueError("数据集为空，无法创建数据加载器，请检查数据加载过程")
    
    # 计算划分大小，确保每个子集至少有1个样本
    train_size = max(1, int(train_ratio * dataset_size))
    val_size = max(1, int(val_ratio * dataset_size))
    
    # 确保三个子集的总和等于dataset_size
    test_size = dataset_size - train_size - val_size
    if test_size <= 0:
        test_size = 1
        if val_size > 1:
            val_size -= 1
        else:
            train_size -= 1
    
    logger.info(f"数据集划分: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")
    
    # 划分数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """
    自定义的收集函数，用于处理不同长度的序列
    
    Args:
        batch: 批次数据
    
    Returns:
        batch_dict: 收集后的批次字典
    """
    # 提取各部分数据
    pdb_ids = [item['pdb_id'] for item in batch]
    chain_ids = [item['chain_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    
    # 确定最大序列长度
    max_len = max(len(item['tokens']) for item in batch)
    
    # 填充tokens到相同长度
    tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        seq_len = len(item['tokens'])
        tokens[i, :seq_len] = item['tokens']
    
    # 合并角度和掩码
    angles = {}
    masks = {}
    
    # 获取所有可能的角度类型
    all_angle_types = set()
    for item in batch:
        all_angle_types.update(item['angles'].keys())
    
    # 对每种角度类型，创建一个张量
    for angle_type in all_angle_types:
        # 确定此角度类型的最大长度
        max_angle_len = max(len(item['angles'].get(angle_type, [])) for item in batch)
        
        # 初始化角度和掩码张量
        angle_tensor = torch.zeros((len(batch), max_angle_len), dtype=torch.float)
        mask_tensor = torch.zeros((len(batch), max_angle_len), dtype=torch.float)
        
        # 填充角度和掩码
        for i, item in enumerate(batch):
            if angle_type in item['angles']:
                angle_data = item['angles'][angle_type]
                mask_data = item['masks'][angle_type]
                angle_len = len(angle_data)
                
                angle_tensor[i, :angle_len] = angle_data
                mask_tensor[i, :angle_len] = mask_data
        
        angles[angle_type] = angle_tensor
        masks[angle_type] = mask_tensor
    
    return {
        'pdb_ids': pdb_ids,
        'chain_ids': chain_ids,
        'sequences': sequences,
        'tokens': tokens,
        'angles': angles,
        'masks': masks
    }