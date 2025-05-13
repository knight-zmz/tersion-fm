# data/adapters.py
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

def compute_torsion_angles_single(atom_coords_dict, sorted_residue_ids):
    """计算单个RNA的扭转角
    
    Args:
        atom_coords_dict: 原子坐标字典，键为残基ID，值为包含'residue_name'和'atom_coords'的字典
        sorted_residue_ids: 排序后的残基ID列表
    
    Returns:
        torsion_angles: 字典，键为角度名，值为角度列表
        torsion_masks: 字典，键为角度名，值为掩码列表（1=有效值，0=缺失值）
    """
    # 定义扭转角所需的原子
    torsion_definitions = {
        # 标准扭转角
        'alpha': ["O3'", 'P', "O5'", "C5'"],    # O3'(i-1)-P(i)-O5'(i)-C5'(i)
        'beta': ['P', "O5'", "C5'", "C4'"],     # P(i)-O5'(i)-C5'(i)-C4'(i)
        'gamma': ["O5'", "C5'", "C4'", "C3'"],  # O5'(i)-C5'(i)-C4'(i)-C3'(i)
        'delta': ["C5'", "C4'", "C3'", "O3'"],  # C5'(i)-C4'(i)-C3'(i)-O3'(i)
        'epsilon': ["C4'", "C3'", "O3'", 'P'],  # C4'(i)-C3'(i)-O3'(i)-P(i+1)
        'zeta': ["C3'", "O3'", 'P', "O5'"],     # C3'(i)-O3'(i)-P(i+1)-O5'(i+1)
        'chi': ["O4'", "C1'", 'N9', 'C4'],      # O4'(i)-C1'(i)-N9(i)-C4(i) (嘌呤)
        
        # 赝扭转角
        'eta': ["C4'", 'P', "C4'", 'P'],        # C4'(i-1)-P(i)-C4'(i)-P(i+1)
        'theta': ['P', "C4'", 'P', "C4'"],      # P(i)-C4'(i)-P(i+1)-C4'(i+1)
        "eta'": ["C1'", 'P', "C1'", 'P'],       # C1'(i-1)-P(i)-C1'(i)-P(i+1)
        "theta'": ['P', "C1'", 'P', "C1'"],     # P(i)-C1'(i)-P(i+1)-C1'(i+1)
        
        # 核糖构象角
        'v0': ["C4'", "O4'", "C1'", "C2'"],     # C4'(i)-O4'(i)-C1'(i)-C2'(i)
        'v1': ["O4'", "C1'", "C2'", "C3'"],     # O4'(i)-C1'(i)-C2'(i)-C3'(i)
        'v2': ["C1'", "C2'", "C3'", "C4'"],     # C1'(i)-C2'(i)-C3'(i)-C4'(i)
        'v3': ["C2'", "C3'", "C4'", "O4'"],     # C2'(i)-C3'(i)-C4'(i)-O4'(i)
        'v4': ["C3'", "C4'", "O4'", "C1'"],     # C3'(i)-C4'(i)-O4'(i)-C1'(i)
    }
    
    # 嘧啶的chi角替代定义
    chi_pyrimidine = ["O4'", "C1'", 'N1', 'C2']  # O4'(i)-C1'(i)-N1(i)-C2(i) (嘧啶)
    
    # 初始化结果
    torsion_angles = {angle_name: [] for angle_name in torsion_definitions}
    torsion_masks = {angle_name: [] for angle_name in torsion_definitions}
    
    # 对每个残基计算扭转角
    for i, res_id in enumerate(sorted_residue_ids):
        # 当前残基的原子坐标
        current_res = atom_coords_dict.get(res_id, {})
        
        # 获取当前残基的碱基类型(如果可用)
        residue_name = None
        if 'residue_name' in current_res:
            residue_name = current_res['residue_name']
            current_res_atoms = current_res['atom_coords']
        else:
            current_res_atoms = current_res
        
        # 计算各种扭转角
        for angle_name, atom_names in torsion_definitions.items():
            angle = None
            mask = 0
            
            try:
                # 根据角度类型选择相应的原子
                if angle_name == 'alpha' and i > 0:
                    # alpha需要前一个残基的O3'
                    prev_res_id = sorted_residue_ids[i-1]
                    prev_res = atom_coords_dict.get(prev_res_id, {})
                    
                    if 'residue_name' in prev_res:
                        prev_res_atoms = prev_res['atom_coords']
                    else:
                        prev_res_atoms = prev_res
                    
                    if "O3'" in prev_res_atoms and all(atom in current_res_atoms for atom in atom_names[1:]):
                        p1 = prev_res_atoms["O3'"]
                        p2 = current_res_atoms['P']
                        p3 = current_res_atoms["O5'"]
                        p4 = current_res_atoms["C5'"]
                        angle = safe_calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                
                elif angle_name == 'epsilon' and i < len(sorted_residue_ids) - 1:
                    # epsilon需要下一个残基的P
                    next_res_id = sorted_residue_ids[i+1]
                    next_res = atom_coords_dict.get(next_res_id, {})
                    
                    if 'residue_name' in next_res:
                        next_res_atoms = next_res['atom_coords']
                    else:
                        next_res_atoms = next_res
                    
                    if 'P' in next_res_atoms and all(atom in current_res_atoms for atom in atom_names[:3]):
                        p1 = current_res_atoms["C4'"]
                        p2 = current_res_atoms["C3'"]
                        p3 = current_res_atoms["O3'"]
                        p4 = next_res_atoms['P']
                        angle = safe_calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                
                elif angle_name == 'zeta' and i < len(sorted_residue_ids) - 1:
                    # zeta需要下一个残基的P和O5'
                    next_res_id = sorted_residue_ids[i+1]
                    next_res = atom_coords_dict.get(next_res_id, {})
                    
                    if 'residue_name' in next_res:
                        next_res_atoms = next_res['atom_coords']
                    else:
                        next_res_atoms = next_res
                    
                    if all(atom in next_res_atoms for atom in atom_names[2:]) and all(atom in current_res_atoms for atom in atom_names[:2]):
                        p1 = current_res_atoms["C3'"]
                        p2 = current_res_atoms["O3'"]
                        p3 = next_res_atoms['P']
                        p4 = next_res_atoms["O5'"]
                        angle = safe_calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                
                elif angle_name == 'chi':
                    # 根据碱基类型选择chi角度定义
                    if residue_name in ['A', 'G']:  # 嘌呤
                        if all(atom in current_res_atoms for atom in ["O4'", "C1'", 'N9', 'C4']):
                            p1 = current_res_atoms["O4'"]
                            p2 = current_res_atoms["C1'"]
                            p3 = current_res_atoms['N9']
                            p4 = current_res_atoms['C4']
                            angle = safe_calculate_dihedral(p1, p2, p3, p4)
                            if angle is not None:
                                mask = 1
                    
                    elif residue_name in ['C', 'U']:  # 嘧啶
                        if all(atom in current_res_atoms for atom in ["O4'", "C1'", 'N1', 'C2']):
                            p1 = current_res_atoms["O4'"]
                            p2 = current_res_atoms["C1'"]
                            p3 = current_res_atoms['N1']
                            p4 = current_res_atoms['C2']
                            angle = safe_calculate_dihedral(p1, p2, p3, p4)
                            if angle is not None:
                                mask = 1
                
                # 处理赝扭转角
                elif angle_name == 'eta' and i > 0 and i < len(sorted_residue_ids) - 1:
                    # eta需要前一个残基的C4'和下一个残基的P
                    prev_res_id = sorted_residue_ids[i-1]
                    next_res_id = sorted_residue_ids[i+1]
                    prev_res = atom_coords_dict.get(prev_res_id, {})
                    next_res = atom_coords_dict.get(next_res_id, {})
                    
                    if 'residue_name' in prev_res:
                        prev_res_atoms = prev_res['atom_coords']
                    else:
                        prev_res_atoms = prev_res
                        
                    if 'residue_name' in next_res:
                        next_res_atoms = next_res['atom_coords']
                    else:
                        next_res_atoms = next_res
                    
                    if "C4'" in prev_res_atoms and 'P' in current_res_atoms and "C4'" in current_res_atoms and 'P' in next_res_atoms:
                        p1 = prev_res_atoms["C4'"]
                        p2 = current_res_atoms['P']
                        p3 = current_res_atoms["C4'"]
                        p4 = next_res_atoms['P']
                        angle = safe_calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                
                elif angle_name == 'theta' and i < len(sorted_residue_ids) - 2:
                    # theta需要当前残基的P和C4'，以及下一个残基的P和下下个残基的C4'
                    if i + 2 < len(sorted_residue_ids):
                        next_res_id = sorted_residue_ids[i+1]
                        next_next_res_id = sorted_residue_ids[i+2]
                        next_res = atom_coords_dict.get(next_res_id, {})
                        next_next_res = atom_coords_dict.get(next_next_res_id, {})
                        
                        if 'residue_name' in next_res:
                            next_res_atoms = next_res['atom_coords']
                        else:
                            next_res_atoms = next_res
                            
                        if 'residue_name' in next_next_res:
                            next_next_res_atoms = next_next_res['atom_coords']
                        else:
                            next_next_res_atoms = next_next_res
                        
                        if 'P' in current_res_atoms and "C4'" in current_res_atoms and 'P' in next_res_atoms and "C4'" in next_next_res_atoms:
                            p1 = current_res_atoms['P']
                            p2 = current_res_atoms["C4'"]
                            p3 = next_res_atoms['P']
                            p4 = next_next_res_atoms["C4'"]
                            angle = safe_calculate_dihedral(p1, p2, p3, p4)
                            if angle is not None:
                                mask = 1
                
                elif angle_name == "eta'" and i > 0 and i < len(sorted_residue_ids) - 1:
                    # eta'需要前一个残基的C1'和下一个残基的P
                    prev_res_id = sorted_residue_ids[i-1]
                    next_res_id = sorted_residue_ids[i+1]
                    prev_res = atom_coords_dict.get(prev_res_id, {})
                    next_res = atom_coords_dict.get(next_res_id, {})
                    
                    if 'residue_name' in prev_res:
                        prev_res_atoms = prev_res['atom_coords']
                    else:
                        prev_res_atoms = prev_res
                        
                    if 'residue_name' in next_res:
                        next_res_atoms = next_res['atom_coords']
                    else:
                        next_res_atoms = next_res
                    
                    if "C1'" in prev_res_atoms and 'P' in current_res_atoms and "C1'" in current_res_atoms and 'P' in next_res_atoms:
                        p1 = prev_res_atoms["C1'"]
                        p2 = current_res_atoms['P']
                        p3 = current_res_atoms["C1'"]
                        p4 = next_res_atoms['P']
                        angle = safe_calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                
                elif angle_name == "theta'" and i < len(sorted_residue_ids) - 2:
                    # theta'需要当前残基的P和C1'，以及下一个残基的P和下下个残基的C1'
                    if i + 2 < len(sorted_residue_ids):
                        next_res_id = sorted_residue_ids[i+1]
                        next_next_res_id = sorted_residue_ids[i+2]
                        next_res = atom_coords_dict.get(next_res_id, {})
                        next_next_res = atom_coords_dict.get(next_next_res_id, {})
                        
                        if 'residue_name' in next_res:
                            next_res_atoms = next_res['atom_coords']
                        else:
                            next_res_atoms = next_res
                            
                        if 'residue_name' in next_next_res:
                            next_next_res_atoms = next_next_res['atom_coords']
                        else:
                            next_next_res_atoms = next_next_res
                        
                        if 'P' in current_res_atoms and "C1'" in current_res_atoms and 'P' in next_res_atoms and "C1'" in next_next_res_atoms:
                            p1 = current_res_atoms['P']
                            p2 = current_res_atoms["C1'"]
                            p3 = next_res_atoms['P']
                            p4 = next_next_res_atoms["C1'"]
                            angle = safe_calculate_dihedral(p1, p2, p3, p4)
                            if angle is not None:
                                mask = 1
                
                # 核糖构象角处理 (v0-v4都只需要当前残基的原子)
                elif angle_name in ['v0', 'v1', 'v2', 'v3', 'v4']:
                    if all(atom in current_res_atoms for atom in atom_names):
                        p1 = current_res_atoms[atom_names[0]]
                        p2 = current_res_atoms[atom_names[1]]
                        p3 = current_res_atoms[atom_names[2]]
                        p4 = current_res_atoms[atom_names[3]]
                        angle = safe_calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                
                else:  # beta, gamma, delta等只需要当前残基的原子
                    if all(atom in current_res_atoms for atom in atom_names):
                        p1 = current_res_atoms[atom_names[0]]
                        p2 = current_res_atoms[atom_names[1]]
                        p3 = current_res_atoms[atom_names[2]]
                        p4 = current_res_atoms[atom_names[3]]
                        angle = safe_calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                
            except Exception as e:
                logger.debug(f"计算{angle_name}角度失败: {str(e)}")
                angle = None
                mask = 0
            
            # 将结果添加到列表，确保不包含NaN值
            torsion_angles[angle_name].append(angle if angle is not None else 0.0)
            torsion_masks[angle_name].append(mask)
    
    return torsion_angles, torsion_masks

def safe_calculate_dihedral(p1, p2, p3, p4):
    """
    安全计算四个原子之间的二面角，带有额外的健壮性检查
    
    Args:
        p1, p2, p3, p4: 四个原子的坐标
    
    Returns:
        dihedral: 二面角（度），如果计算失败则返回None
    """
    try:
        # 转换为numpy数组
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        p3 = np.array(p3, dtype=np.float64)
        p4 = np.array(p4, dtype=np.float64)
        
        # 检查坐标是否有效
        if (np.isnan(p1).any() or np.isnan(p2).any() or 
            np.isnan(p3).any() or np.isnan(p4).any() or
            np.isinf(p1).any() or np.isinf(p2).any() or
            np.isinf(p3).any() or np.isinf(p4).any()):
            return None
        
        # 计算向量
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        # 检查向量是否为零向量
        b1_norm = np.linalg.norm(b1)
        b2_norm = np.linalg.norm(b2)
        b3_norm = np.linalg.norm(b3)
        
        if b1_norm < 1e-6 or b2_norm < 1e-6 or b3_norm < 1e-6:
            return None
        
        # 归一化向量b2
        b2_unit = b2 / b2_norm
        
        # 计算法向量
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # 检查法向量是否为零向量
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        
        if n1_norm < 1e-6 or n2_norm < 1e-6:
            return None
        
        # 安全归一化
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        
        # 计算平面向量
        m1 = np.cross(n1, b2_unit)
        
        # 计算二面角
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        # 限制点积结果在[-1,1]范围内，避免数值误差
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        
        # 计算角度（弧度）
        dihedral = np.arctan2(y, x)
        
        # 转换为度
        dihedral_degrees = np.degrees(dihedral)
        
        # 最终检查结果
        if np.isnan(dihedral_degrees) or np.isinf(dihedral_degrees):
            return None
            
        return float(dihedral_degrees)
    
    except Exception as e:
        logger.debug(f"二面角计算失败: {str(e)}")
        return None
    

def adapt_training_dict_single(data):
    """
    适配Training_Dict_single.pkl格式的数据
    
    参数:
        data: 原始数据字典，键为结构ID，值为RNA结构信息
    
    返回:
        list: 转换后的数据列表，符合模型预期的格式
    """
    adapted_data = []
    
    # 处理每个RNA结构
    for key, item in data.items():
        try:
            # 检查格式是否正确
            if not isinstance(item, dict) or 'rna_dic' not in item:
                logger.warning(f"结构 {key} 不符合预期格式，跳过")
                continue
            
            # 提取基本信息
            pdb_id = item.get('pdb_id', f'unknown_{key}')
            chain_id = item.get('chain_id', 'A')
            is_multi_chain = item.get('if_multi_chain', False)
            
            # 跳过多链RNA
            if is_multi_chain:
                logger.info(f"跳过多链RNA: {pdb_id}")
                continue
            
            # 获取RNA字典
            rna_dic = item['rna_dic']
            if not isinstance(rna_dic, dict) or len(rna_dic) == 0:
                logger.warning(f"RNA字典为空或格式无效: {pdb_id}，跳过")
                continue
            
            # 排序残基ID确保序列顺序正确
            sorted_residue_ids = sorted(rna_dic.keys())
            
            # 提取序列和原子坐标
            sequence = ""
            for res_id in sorted_residue_ids:
                residue = rna_dic[res_id]
                
                # 检查残基格式
                if not isinstance(residue, dict) or 'residue_name' not in residue:
                    logger.warning(f"残基 {res_id} 在 {pdb_id} 中格式无效，跳过")
                    continue
                
                residue_name = residue['residue_name']
                sequence += residue_name
            
            # 计算扭转角
            torsion_angles, torsion_masks = compute_torsion_angles_single(rna_dic, sorted_residue_ids)
            
            # 创建适配后的结构
            adapted_item = {
                'pdb_id': pdb_id,
                'chain_id': chain_id,
                'sequence': sequence,
                'torsion_angles': torsion_angles,
                'torsion_masks': torsion_masks,
                'sorted_residue_ids': sorted_residue_ids
            }
            
            adapted_data.append(adapted_item)
            logger.debug(f"成功处理结构: {pdb_id}, 序列长度: {len(sequence)}")
            
        except Exception as e:
            logger.warning(f"处理结构 {key} 时出错: {str(e)}")
    
    logger.info(f"适配转换: 原始数据 {len(data)} 项 -> 适配后 {len(adapted_data)} 项")
    return adapted_data