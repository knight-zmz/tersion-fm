# data/preprocessing.py
"""
数据预处理模块：从pkl文件提取原子坐标，计算扭转角
"""

import pickle
import numpy as np
from collections import defaultdict
import os
import logging

logger = logging.getLogger(__name__)

def load_pdb_data(pkl_path):
    """
    从pkl文件加载RNA数据
    
    Args:
        pkl_path: pkl文件路径
    
    Returns:
        chain_id: 链ID
        sequence: RNA序列
        atom_coords_dict: 原子坐标字典
        is_multi_chain: 是否为多链RNA
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 检查是否多链RNA，我们只处理单链RNA
        if data.get('if_multi_chain', False):
            logger.info(f"跳过多链RNA: {pkl_path}")
            return None, None, None, True
        
        chain_id = data.get('chain_id', 'A')
        pdb_id = data.get('pdb_id', os.path.basename(pkl_path).split('.')[0])
        rna_dic = data.get('rna_dic', {})
        
        # 提取序列和原子坐标
        sequence = ""
        atom_coords_dict = {}
        
        # 按照键的顺序排序残基
        sorted_keys = sorted(rna_dic.keys())
        
        for key in sorted_keys:
            residue = rna_dic[key]
            residue_name = residue.get('residue_name', '')
            if residue_name:
                sequence += residue_name
                atom_coords_dict[key] = residue.get('atom_coords', {})
        
        return chain_id, sequence, atom_coords_dict, False
        
    except Exception as e:
        logger.error(f"加载文件失败 {pkl_path}: {str(e)}")
        return None, None, None, None

def calculate_dihedral(p1, p2, p3, p4):
    """
    计算四个原子之间的二面角
    
    Args:
        p1, p2, p3, p4: 四个原子的坐标
    
    Returns:
        dihedral: 二面角（度）
    """
    try:
        # 转换为numpy数组
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        p4 = np.array(p4)
        
        # 计算向量
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        # 归一化向量b2
        b2_norm = b2 / np.linalg.norm(b2)
        
        # 计算法向量
        n1 = np.cross(b1, b2)
        n1 = n1 / np.linalg.norm(n1)
        n2 = np.cross(b2, b3)
        n2 = n2 / np.linalg.norm(n2)
        
        # 计算平面向量
        m1 = np.cross(n1, b2_norm)
        
        # 计算二面角
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        # 计算角度（弧度）
        dihedral = np.arctan2(y, x)
        
        # 转换为度
        dihedral_degrees = np.degrees(dihedral)
        
        return dihedral_degrees
    except Exception as e:
        logger.warning(f"计算二面角失败: {str(e)}")
        return None

def compute_torsion_angles(atom_coords_dict, sorted_residue_ids):
    """
    计算RNA扭转角
    
    Args:
        atom_coords_dict: 原子坐标字典，键为残基ID，值为原子坐标字典
        sorted_residue_ids: 排序后的残基ID列表
    
    Returns:
        torsion_angles: 字典，键为角度名，值为角度列表
        torsion_masks: 字典，键为角度名，值为掩码列表（1=有效值，0=缺失值）
    """
    # 定义扭转角所需的原子
    torsion_definitions = {
        'alpha': ["O3'", 'P', "O5'", "C5'"],    # O3'(i-1)-P(i)-O5'(i)-C5'(i)
        'beta': ['P', "O5'", "C5'", "C4'"],     # P(i)-O5'(i)-C5'(i)-C4'(i)
        'gamma': ["O5'", "C5'", "C4'", "C3'"],  # O5'(i)-C5'(i)-C4'(i)-C3'(i)
        'delta': ["C5'", "C4'", "C3'", "O3'"],  # C5'(i)-C4'(i)-C3'(i)-O3'(i)
        'epsilon': ["C4'", "C3'", "O3'", 'P'],  # C4'(i)-C3'(i)-O3'(i)-P(i+1)
        'zeta': ["C3'", "O3'", 'P', "O5'"],     # C3'(i)-O3'(i)-P(i+1)-O5'(i+1)
        'chi': ["O4'", "C1'", 'N9', 'C4']       # O4'(i)-C1'(i)-N9(i)-C4(i) (purine)
        # 注：对于嘧啶，chi为O4'(i)-C1'(i)-N1(i)-C2(i)
    }
    
    # 初始化结果
    torsion_angles = {angle_name: [] for angle_name in torsion_definitions}
    torsion_masks = {angle_name: [] for angle_name in torsion_definitions}
    
    # 对每个残基计算扭转角
    for i, res_id in enumerate(sorted_residue_ids):
        # 当前残基的原子坐标
        current_res_atoms = atom_coords_dict.get(res_id, {})
        
        # 计算各种扭转角
        for angle_name, atom_names in torsion_definitions.items():
            angle = None
            mask = 0
            
            # 根据角度类型选择相应的原子
            if angle_name == 'alpha' and i > 0:
                # alpha需要前一个残基的O3'
                prev_res_id = sorted_residue_ids[i-1]
                prev_res_atoms = atom_coords_dict.get(prev_res_id, {})
                
                if "O3'" in prev_res_atoms and all(atom in current_res_atoms for atom in atom_names[1:]):
                    try:
                        p1 = prev_res_atoms["O3'"]
                        p2 = current_res_atoms['P']
                        p3 = current_res_atoms["O5'"]
                        p4 = current_res_atoms["C5'"]
                        angle = calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                    except Exception as e:
                        logger.debug(f"计算alpha角度失败: {str(e)}")
            
            elif angle_name == 'epsilon' and i < len(sorted_residue_ids) - 1:
                # epsilon需要下一个残基的P
                next_res_id = sorted_residue_ids[i+1]
                next_res_atoms = atom_coords_dict.get(next_res_id, {})
                
                if 'P' in next_res_atoms and all(atom in current_res_atoms for atom in atom_names[:3]):
                    try:
                        p1 = current_res_atoms["C4'"]
                        p2 = current_res_atoms["C3'"]
                        p3 = current_res_atoms["O3'"]
                        p4 = next_res_atoms['P']
                        angle = calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                    except Exception as e:
                        logger.debug(f"计算epsilon角度失败: {str(e)}")
            
            elif angle_name == 'zeta' and i < len(sorted_residue_ids) - 1:
                # zeta需要下一个残基的P和O5'
                next_res_id = sorted_residue_ids[i+1]
                next_res_atoms = atom_coords_dict.get(next_res_id, {})
                
                if all(atom in next_res_atoms for atom in atom_names[2:]) and all(atom in current_res_atoms for atom in atom_names[:2]):
                    try:
                        p1 = current_res_atoms["C3'"]
                        p2 = current_res_atoms["O3'"]
                        p3 = next_res_atoms['P']
                        p4 = next_res_atoms["O5'"]
                        angle = calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                    except Exception as e:
                        logger.debug(f"计算zeta角度失败: {str(e)}")
            
            elif angle_name == 'chi':
                # chi角度根据碱基类型选择不同的原子
                residue_name = atom_coords_dict[res_id].get('residue_name', '')
                
                if residue_name in ['A', 'G']:  # 嘌呤
                    if all(atom in current_res_atoms for atom in ["O4'", "C1'", 'N9', 'C4']):
                        try:
                            p1 = current_res_atoms["O4'"]
                            p2 = current_res_atoms["C1'"]
                            p3 = current_res_atoms['N9']
                            p4 = current_res_atoms['C4']
                            angle = calculate_dihedral(p1, p2, p3, p4)
                            if angle is not None:
                                mask = 1
                        except Exception as e:
                            logger.debug(f"计算chi角度(嘌呤)失败: {str(e)}")
                
                elif residue_name in ['C', 'U']:  # 嘧啶
                    if all(atom in current_res_atoms for atom in ["O4'", "C1'", 'N1', 'C2']):
                        try:
                            p1 = current_res_atoms["O4'"]
                            p2 = current_res_atoms["C1'"]
                            p3 = current_res_atoms['N1']
                            p4 = current_res_atoms['C2']
                            angle = calculate_dihedral(p1, p2, p3, p4)
                            if angle is not None:
                                mask = 1
                        except Exception as e:
                            logger.debug(f"计算chi角度(嘧啶)失败: {str(e)}")
            
            else:  # beta, gamma, delta等只需要当前残基的原子
                if all(atom in current_res_atoms for atom in atom_names):
                    try:
                        p1 = current_res_atoms[atom_names[0]]
                        p2 = current_res_atoms[atom_names[1]]
                        p3 = current_res_atoms[atom_names[2]]
                        p4 = current_res_atoms[atom_names[3]]
                        angle = calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                    except Exception as e:
                        logger.debug(f"计算{angle_name}角度失败: {str(e)}")
            
            # 将结果添加到列表
            torsion_angles[angle_name].append(angle if angle is not None else 0.0)
            torsion_masks[angle_name].append(mask)
    
    return torsion_angles, torsion_masks

def process_pdb_file(pkl_path):
    """
    处理单个PDB文件，提取序列和计算扭转角
    
    Args:
        pkl_path: pkl文件路径
    
    Returns:
        result_dict: 包含处理结果的字典
    """
    # 加载数据
    chain_id, sequence, atom_coords_dict, is_multi_chain = load_pdb_data(pkl_path)
    
    # 如果是多链RNA或加载失败，则返回None
    if is_multi_chain or sequence is None:
        return None
    
    # 排序的残基ID列表
    sorted_residue_ids = sorted(atom_coords_dict.keys())
    
    # 计算扭转角
    torsion_angles, torsion_masks = compute_torsion_angles(atom_coords_dict, sorted_residue_ids)
    
    # 构建结果字典
    result_dict = {
        'pdb_id': os.path.basename(pkl_path).split('.')[0],
        'chain_id': chain_id,
        'sequence': sequence,
        'torsion_angles': torsion_angles,
        'torsion_masks': torsion_masks,
        'sorted_residue_ids': sorted_residue_ids
    }
    
    return result_dict