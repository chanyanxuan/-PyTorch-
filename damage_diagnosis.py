"""
功能：
1. 从CSV文件读取有限元模拟数据和实际损伤案例进行训练和验证
2. 基于InSAR数据反演建筑物损伤情况
输入文件：
- train.csv：训练数据（有限元批量仿真的损伤情况+工况）
- test_building.csv：待推理数据（InSAR得到的实际建筑物屋面损伤情况）
输出文件：
1. damage_model.pt：训练后的模型权重
2. training_log.csv：训练日志，记录每个epoch的损失、准确率和误差
3. test_building_pred.json：推理结果，包含预测的象限编号、损伤程度和置信度
日期：2025-10-15
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建筑物损伤诊断系统

功能：
1. 从CSV文件读取有限元模拟数据和实际损伤案例进行训练和验证
2. 基于InSAR数据反演建筑物损伤情况

输入文件：
- train.csv：训练数据（有限元批量仿真的损伤情况+工况）
- test_building.csv：待推理数据（InSAR得到的实际建筑物屋面损伤情况）

输出文件：
1. damage_model.pt：训练后的模型权重
2. training_log.csv：训练日志，记录每个epoch的损失、准确率和误差
3. test_building_pred.json：推理结果，包含预测的象限编号、损伤程度和置信度

日期：2025-10-15
版本：2.0 (优化版)
"""

# 标准库导入
import argparse
import csv
import json
import logging
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# 第三方库导入
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import Dataset, DataLoader, random_split
    from tqdm import tqdm
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"缺少必要的依赖库: {e}")
    print("请运行: pip install torch numpy pandas tqdm matplotlib")
    exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('damage_diagnosis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================== 配置类 ====================
class Config:
    """配置类，集中管理所有超参数和设置"""

    # 模型参数
    N_QUADRANT: int = 3  # 象限数量
    HIDDEN_DIM: int = 128  # 隐藏层维度
    DROPOUT_RATE: float = 0.2  # Dropout比率

    # 训练参数
    SEED: int = 42  # 随机种子
    EPOCHS: int = 1000  # 训练轮数
    LR: float = 1e-4  # 学习率
    WD: float = 1e-5  # 权重衰减
    PATIENCE: int = 100  # 早停耐心值
    SAVE_EVERY: int = 50  # 每隔多少轮保存一次检查点
    BATCH_SIZE: Optional[int] = None  # 批大小（None表示自动计算）

    # 文件路径
    MODEL_DIR: str = "models"  # 模型保存目录
    LOG_FILE: str = "training_log.csv"  # 训练日志文件
    BEST_METRIC: str = "val_loss"  # 最佳模型评估指标

    # 设备配置
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    NORMALIZE: bool = True  # 是否标准化
    SCALE_SLOPE: float = 1.0  # 斜率缩放因子

    # 填充策略
    PADDING_STRATEGY: str = 'mask'  # 填充策略（默认使用mask+0填充以配合注意力的key_padding_mask）

    # 注意力与位置编码
    MAX_SEQ_LEN: int = 8192  # 学习式位置编码的最大支持序列长度（超出将自动使用正弦位置编码）
    POS_ENCODING: str = 'auto'  # 'auto'|'learned'|'sinusoidal'
    ATTN_MAX_LEN: int = 4096    # 注意力参与的最大序列长度（超出将自动跳过注意力）

    @property
    def model_path(self) -> str:
        """模型保存路径"""
        return os.path.join(self.MODEL_DIR, "damage_model.pt")

    @property
    def log_path(self) -> str:
        """日志文件路径"""
        return self.LOG_FILE

# 全局配置实例
config = Config()

N_QUADRANT = config.N_QUADRANT
MODEL_DIR = config.MODEL_DIR
MODEL_PATH = config.model_path
DEVICE = config.DEVICE
SEED = config.SEED
EPOCHS = config.EPOCHS
LR = config.LR
WD = config.WD
PATIENCE = config.PATIENCE
SAVE_EVERY = config.SAVE_EVERY
LOG_FILE = config.log_path
BEST_METRIC = config.BEST_METRIC

def set_seed(seed: int = config.SEED) -> None:
    """设置随机种子以确保结果可重现

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    logger.info(f"随机种子设置为: {seed}")


# 数据集类
class BuildingDamageDataset(Dataset):
    """建筑物损伤数据集类

    功能：
    1. 读取CSV文件并解析数据
    2. 对数据进行标准化处理
    3. 支持训练和测试模式

    Args:
        csv_path: CSV文件路径
        has_labels: 数据是否包含标签
        normalize: 是否进行标准化
        scale_slope: 斜率缩放因子
        is_test: 是否为测试模式
    """

    def __init__(self, csv_path: str, has_labels: bool = True, normalize: bool = True,
                 scale_slope: float = 1.0, is_test: bool = False) -> None:
        self.has_labels = has_labels
        self.normalize = normalize
        self.scale_slope = scale_slope
        self.is_test = is_test

        # 读取CSV文件
        rows = []
        try:
            # 使用 utf-8-sig 以自动移除BOM，避免如 "\ufeff7000.0" 的转换失败
            with open(csv_path, newline='', encoding='utf-8-sig') as f:
                rdr = csv.reader(f)
                for line_num, line in enumerate(rdr, 1):
                    if not line:  # 跳过空行
                        continue
                    try:
                        # 转换为浮点数，先strip并移除可能混入的BOM
                        float_line = [float(str(v).strip().lstrip('\ufeff')) for v in line]
                        rows.append(float_line)
                    except ValueError as e:
                        logger.warning(f"第{line_num}行数据转换失败: {e}, 跳过该行")
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV文件未找到: {csv_path}")
        except Exception as e:
            raise RuntimeError(f"读取CSV文件时发生错误: {e}")

        if not rows:
            raise ValueError(f"CSV文件为空或没有有效数据: {csv_path}")

        # 验证数据格式
        if has_labels and any(len(r) < 4 for r in rows):
            raise ValueError("CSV 至少需要 4 列 (features..., span, width, severity, loc)")
        elif not has_labels and any(len(r) < 2 for r in rows):
            raise ValueError("CSV 至少需要 2 列 (features..., span, width)")

        self.samples: List[Tuple[torch.Tensor, ...]] = []
        # 初始化标准化参数
        self.d_mean, self.d_std = 0.0, 1.0  
        self.s_mean, self.s_std = 0.0, 1.0  
        self.span_mean, self.span_std = 0.0, 1.0  
        self.width_mean, self.width_std = 0.0, 1.0  
        self.sev_mean, self.sev_std = 0.0, 1.0  


        if normalize and len(rows) > 0:
            all_d, all_s, all_span, all_width, all_sev = [], [], [], [], []  # 存储所有距离、斜率和损伤程度
            for row in rows:
                # 提取特征（最后一列是位置，倒数第二列是损伤程度）
                feats = row[:-4] if has_labels else row[:-2]
                if len(feats) % 2 == 1:  # 确保特征数量是偶数
                    feats = feats[:-1]
                if len(feats) == 0:  # 跳过空特征
                    continue

                # 提取距离和斜率特征
                d_vec = feats[0::2]  # 距离特征（偶数索引）
                s_vec = feats[1::2]  # 斜率特征（奇数索引）
                span = row[-4] if has_labels else row[-2]
                width = row[-3] if has_labels else row[-1]
                # 测试模式下放大斜率
                if is_test:
                    s_vec = [s * scale_slope for s in s_vec]

                # 收集所有数据用于计算统计量
                all_d.extend(d_vec)
                all_s.extend(s_vec)
                all_span.append(span)
                all_width.append(width)
                if has_labels:
                    sev = float(row[-2])  # 损伤程度
                    all_sev.append(sev)

            if all_d and all_s:
                self.d_mean, self.d_std = np.mean(all_d), np.std(all_d)
                self.s_mean, self.s_std = np.mean(all_s), np.std(all_s)
                self.span_mean, self.span_std = np.mean(all_span), np.std(all_span)
                self.width_mean, self.width_std = np.mean(all_width), np.std(all_width)
                # 避免除零错误
                self.d_std = self.d_std if self.d_std > 0 else 1
                self.s_std = self.s_std if self.s_std > 0 else 1
                self.span_std = self.span_std if self.span_std > 0 else 1
                self.width_std = self.width_std if self.width_std > 0 else 1
            if all_sev:
                self.sev_mean, self.sev_std = np.mean(all_sev), np.std(all_sev)
                self.sev_std = self.sev_std if self.sev_std > 0 else 1

        # 处理每一行数据
        for row in rows:
            if has_labels:
                span = float(row[-4])
                width = float(row[-3])
                sev = float(row[-2])  
                loc = int(row[-1])  
                feats = row[:-4]  
            else:
                feats = row[:-2]
                span, width, sev, loc = row[-2], row[-1], 0.0, 0.0  


            if len(feats) % 2 == 1:
                feats = feats[:-1]
            if len(feats) == 0: 
                continue

            d_vec = feats[0::2]
            s_vec = feats[1::2]

            # 测试模式下放大斜率
            if is_test:
                s_vec = [s * scale_slope for s in s_vec]

            # 标准化数据
            if normalize:
                d_vec = [(d - self.d_mean) / self.d_std for d in d_vec]
                s_vec = [(s - self.s_mean) / self.s_std for s in s_vec]
                span = (span-self.span_mean)/self.span_std
                width = (width-self.width_mean)/self.width_std
                if has_labels:
                    sev = (sev - self.sev_mean) / self.sev_std

            # 存储处理后的样本
            self.samples.append((
                torch.tensor(d_vec, dtype=torch.float32),  
                torch.tensor(s_vec, dtype=torch.float32),  
                torch.tensor(span, dtype=torch.float32),  
                torch.tensor(width, dtype=torch.float32),  
                torch.tensor(loc, dtype=torch.long),  
                torch.tensor(sev, dtype=torch.float32)  
            ))

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        return self.samples[idx]


def collate_fn(batch, padding_strategy='mean'):
    """自定义批处理函数，用于处理可变长度序列
    功能：
    1. 对批次中的可变长度序列进行填充，避免使用0填充
    2. 支持多种填充策略
    参数：
    batch: 批数据
    padding_strategy: 填充策略
        - 'mean': 使用批次内该特征的均值填充
        - 'median': 使用批次内该特征的中位数填充
        - 'last': 使用序列的最后一个值填充
        - 'replicate': 复制最后一个值填充
        - 'no_pad': 不填充，使用pack_padded_sequence处理

    返回：
    填充后的距离、斜率、位置和损伤程度张量
    """
    d, s, span, width, loc, sev = zip(*batch)

    if padding_strategy == 'no_pad':
        # 方案1：不填充，使用pack_padded_sequence
        # 需要修改模型架构以支持可变长度序列,基本等于推倒重来，太麻烦了
        return d, s, torch.stack(loc), torch.stack(sev)

    elif padding_strategy == 'mean':
        # 方案2：使用批次内均值填充（保持向后兼容，不建议与注意力同时使用）
        d_flat = [item for seq in d for item in seq]
        s_flat = [item for seq in s for item in seq]
        d_mean = torch.tensor(d_flat).mean().item()
        s_mean = torch.tensor(s_flat).mean().item()

        d_pad = pad_sequence(d, batch_first=True, padding_value=d_mean).float()
        s_pad = pad_sequence(s, batch_first=True, padding_value=s_mean).float()
        # 均值填充下不构造mask
        key_padding_mask = torch.zeros((len(d), d_pad.size(1)), dtype=torch.bool)

    elif padding_strategy == 'median':
        # 方案3：使用批次内中位数填充
        d_flat = [item for seq in d for item in seq]
        s_flat = [item for seq in s for item in seq]
        d_median = torch.tensor(d_flat).median().item()
        s_median = torch.tensor(s_flat).median().item()

        d_pad = pad_sequence(d, batch_first=True, padding_value=d_median).float()
        s_pad = pad_sequence(s, batch_first=True, padding_value=s_median).float()
        key_padding_mask = torch.zeros((len(d), d_pad.size(1)), dtype=torch.bool)

    elif padding_strategy == 'last':
        # 方案4：使用序列最后一个值填充
        d_pad = pad_sequence(d, batch_first=True, padding_value=0.0).float()
        s_pad = pad_sequence(s, batch_first=True, padding_value=0.0).float()

        # 获取每个序列的实际长度
        d_lengths = [len(seq) for seq in d]
        s_lengths = [len(seq) for seq in s]

        # 用最后一个有效值填充
        for i, (d_len, s_len) in enumerate(zip(d_lengths, s_lengths)):
            if d_len < d_pad.size(1):
                d_pad[i, d_len:] = d_pad[i, d_len - 1]
            if s_len < s_pad.size(1):
                s_pad[i, s_len:] = s_pad[i, s_len - 1]
        key_padding_mask = torch.zeros((len(d), d_pad.size(1)), dtype=torch.bool)

    elif padding_strategy == 'replicate':
        # 方案5：复制最后一个值填充
        d_pad = pad_sequence(d, batch_first=True, padding_value=0.0).float()
        s_pad = pad_sequence(s, batch_first=True, padding_value=0.0).float()

        # 获取每个序列的实际长度
        d_lengths = [len(seq) for seq in d]
        s_lengths = [len(seq) for seq in s]

        # 用最后一个有效值填充
        for i, (d_len, s_len) in enumerate(zip(d_lengths, s_lengths)):
            if d_len < d_pad.size(1):
                d_pad[i, d_len:] = d_pad[i, d_len - 1]
            if s_len < s_pad.size(1):
                s_pad[i, s_len:] = s_pad[i, s_len - 1]
        key_padding_mask = torch.zeros((len(d), d_pad.size(1)), dtype=torch.bool)

    elif padding_strategy == 'mask':
        # 方案6：0填充 + key_padding_mask（推荐用于注意力）
        d_lengths = [len(seq) for seq in d]
        s_lengths = [len(seq) for seq in s]
        assert d_lengths == s_lengths, "d与s长度应一致"
        d_pad = pad_sequence(d, batch_first=True, padding_value=0.0).float()
        s_pad = pad_sequence(s, batch_first=True, padding_value=0.0).float()
        max_len = d_pad.size(1)
        key_padding_mask = torch.zeros((len(d), max_len), dtype=torch.bool)
        for i, L in enumerate(d_lengths):
            if L < max_len:
                key_padding_mask[i, L:] = True  # True表示需要mask掉

    else:
        raise ValueError(f"不支持的填充策略: {padding_strategy}")

    return d_pad, s_pad, torch.stack(span).float(), torch.stack(width).float(), torch.stack(loc), torch.stack(sev), key_padding_mask
def create_collate_fn(padding_strategy='mean'):
    """创建指定填充策略的collate_fn函数
    参数：
    padding_strategy: 填充策略
    返回：
    配置好的collate_fn函数
    """
    def _collate_fn(batch):
        return collate_fn(batch, padding_strategy)
    return _collate_fn


# 神经网络模型
class StructuralDamageNet(nn.Module):
    """建筑物结构损伤诊断神经网络 - 

    特点：
    1. 双流架构同时处理距离和斜率特征
    2. 多任务学习同时预测损伤位置和程度
    3. 自适应特征提取适用于不同尺寸的建筑物
    4. 支持残差连接和注意力机制
    5. 可配置的网络深度和宽度

    Args:
        hidden: 隐藏层维度
        dropout: Dropout比率
        use_attention: 是否使用注意力机制
        use_residual: 是否使用残差连接
    """

    def __init__(self, hidden: int = config.HIDDEN_DIM, dropout: float = config.DROPOUT_RATE,
                 use_attention: bool = True, use_residual: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.use_residual = use_residual

        # 1D卷积网络 - 改进版
        self.conv1 = nn.Conv1d(2, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # 注意力机制
        if use_attention:
            self.attention = nn.MultiheadAttention(256, num_heads=8, dropout=dropout, batch_first=True)
            # 可学习位置编码（最大序列长度由Config指定）
            self.pos_embedding = nn.Embedding(config.MAX_SEQ_LEN, 256)
            self.pos_dropout = nn.Dropout(dropout)
            # 预生成一组正弦位置编码（按需扩展）
            self.register_buffer('sin_cache', torch.zeros(1, 1, 256), persistent=False)

        # 自适应池化
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256 + 2, hidden),  # +2 for span and width
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )

        # 位置预测头（分类任务）
        self.loc_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, config.N_QUADRANT)
        )

        # 损伤程度预测头（回归任务）- 优化版本
        self.sev_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 4, 1)  # 最终输出 1
        )

    def forward(self, d: torch.Tensor, s: torch.Tensor,
                span: torch.Tensor, width: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """前向传播

        Args:
            d: 距离特征 [batch_size, seq_len]
            s: 斜率特征 [batch_size, seq_len]
            span: 跨度特征 [batch_size]
            width: 宽度特征 [batch_size]

        Returns:
            loc_pred: 位置预测（分类）[batch_size, n_quadrant]
            sev_pred: 损伤程度预测（回归）[batch_size]
        """
        # 将距离和斜率堆叠为2通道输入
        x = torch.stack([d, s], dim=1)  # [batch_size, 2, seq_len]

        # 通过卷积网络提取特征
        h1 = torch.relu(self.bn1(self.conv1(x)))
        h2 = torch.relu(self.bn2(self.conv2(h1)))
        h3 = torch.relu(self.bn3(self.conv3(h2)))

        # 残差连接
        if self.use_residual and h2.size() == h3.size():
            h3 = h3 + h2

        # 注意力机制
        if self.use_attention:
            # 转换维度用于注意力计算
            h3_att = h3.transpose(1, 2)  # [batch_size, seq_len, channels]
            # 位置编码（自动策略）：短序列用可学习编码，超长时退回正弦编码
            seq_len = h3_att.size(1)
            if config.POS_ENCODING in ('auto', 'learned') and seq_len <= config.MAX_SEQ_LEN:
                pos = torch.arange(seq_len, device=h3_att.device)
                pos_emb = self.pos_embedding(pos).unsqueeze(0)  # [1, L, 256]
                h3_att = self.pos_dropout(h3_att + pos_emb)
            else:
                # 使用正弦位置编码（支持任意长度）
                pos_emb = self._sinusoidal_pos_emb(seq_len, h3_att.device)  # [1, L, 256]
                h3_att = self.pos_dropout(h3_att + pos_emb)
            # 当序列过长，为避免O(L^2)显存爆炸，自动跳过注意力（仅卷积+池化）
            if seq_len > config.ATTN_MAX_LEN:
                attn_weights = None
            else:
                # 使用key_padding_mask（True位置会被忽略）。常规路径不需要保存权重。
                need_weights = return_attention
                attn_out, attn_weights = self.attention(h3_att, h3_att, h3_att, key_padding_mask=key_padding_mask, need_weights=need_weights)
                h3 = attn_out.transpose(1, 2)  # 转回原维度

        # 自适应池化
        h = self.adaptive_pool(h3).squeeze(-1)  # [batch_size, channels]

        # 拼接几何标量特征 span 与 width
        h = torch.cat([h, span.unsqueeze(1), width.unsqueeze(1)], dim=1)

        # 通过全连接层
        h = self.fc(h)

        # 返回位置预测和损伤程度预测
        if return_attention and self.use_attention:
            return self.loc_head(h), self.sev_head(h).squeeze(-1), attn_weights
        return self.loc_head(h), self.sev_head(h).squeeze(-1)

    def _sinusoidal_pos_emb(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成正弦位置编码，形状 [1, L, 256]。
        与Transformer一致：奇偶维度分别为sin/cos，频率按指数递增。
        """
        d_model = 256
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-(math.log(10000.0) / d_model)))  # [d_model/2]
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1,L,256]


# 训练函数
def train_model(csv_path: str,
                resume: bool = False,
                only_new: bool = False,
                epochs: int = config.EPOCHS,
                lr: float = config.LR,
                wd: float = config.WD,
                patience: int = config.PATIENCE,
                save_every: int = config.SAVE_EVERY,
                padding_strategy: str = config.PADDING_STRATEGY,
                batch_size: Optional[int] = config.BATCH_SIZE,
                num_workers: int = 0,
                amp: bool = False,
                benchmark: bool = False,
                use_attention: bool = True,
                use_residual: bool = True) -> Dict[str, float]:
    """训练模型 - 

    功能亮点：
    1. 支持从检查点恢复训练
    2. 动态学习率调整
    3. 早停机制防止过拟合
    4. 多任务损失平衡
    5. 支持多种填充策略避免使用0填充
    6. 改进的日志记录和性能监控
    7. 支持注意力机制和残差连接

    Args:
        csv_path: 训练数据CSV文件路径
        resume: 是否从检查点恢复训练
        only_new: 是否使用全部数据训练（无验证集）
        epochs: 训练轮数
        lr: 学习率
        wd: 权重衰减
        patience: 早停耐心值
        save_every: 每隔多少轮保存一次检查点
        padding_strategy: 填充策略
        batch_size: 批大小（None表示自动计算）
        num_workers: DataLoader工作进程数
        amp: 是否启用混合精度训练
        benchmark: 是否开启cudnn.benchmark
        use_attention: 是否使用注意力机制
        use_residual: 是否使用残差连接

    """
    set_seed()  # 设置随机种子

    # 记录训练开始
    logger.info(f"开始训练模型，数据文件: {csv_path}")
    logger.info(f"训练参数: epochs={epochs}, lr={lr}, wd={wd}, patience={patience}")
    logger.info(f"模型参数: use_attention={use_attention}, use_residual={use_residual}")

    # 创建模型目录
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 设备与后端设置
    if torch.cuda.is_available():
        if benchmark:
            torch.backends.cudnn.benchmark = True
        logger.info(f"使用设备: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        logger.info("使用设备: CPU")

    # 加载数据集
    try:
        ds = BuildingDamageDataset(csv_path, has_labels=True, normalize=True)
        logger.info(f"成功加载数据集，样本数量: {len(ds)}")
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise

    # 动态计算批次大小
    BATCH_SIZE = batch_size if batch_size and batch_size > 0 else min(16, max(2, len(ds) // 10 + 1))
    logger.info(f"使用批大小: {BATCH_SIZE}")

    # 划分训练集和验证集
    if only_new or len(ds) < 10:
        train_ds = val_ds = ds
        print(f"[INFO] 使用全部 {len(ds)} 个样本进行训练和验证")
    else:
        split = int(0.8 * len(ds))
        train_ds, val_ds = random_split(ds, [split, len(ds) - split])
        print(f"[INFO] 数据集分割: {len(train_ds)} 训练, {len(val_ds)} 验证")

    # 创建数据加载器
    pin = True if DEVICE.type == 'cuda' else False
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=create_collate_fn(padding_strategy), num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                            collate_fn=create_collate_fn(padding_strategy), num_workers=num_workers, pin_memory=pin)

    print(f"[INFO] 使用填充策略: {padding_strategy}")
    if padding_strategy == 'mean':
        print("[INFO] 使用批次内均值填充，避免0填充对模型的影响")
    elif padding_strategy == 'median':
        print("[INFO] 使用批次内中位数填充，对异常值更鲁棒")
    elif padding_strategy == 'last':
        print("[INFO] 使用序列最后一个值填充，保持数据连续性")
    elif padding_strategy == 'replicate':
        print("[INFO] 复制最后一个值填充，适合时间序列数据")
    elif padding_strategy == 'no_pad':
        print("[INFO] 不进行填充，需要修改模型架构支持可变长度序列")
    elif padding_strategy == 'mask':
        print("[INFO] 使用0填充+key_padding_mask，注意力将忽略填充位置")

    # 初始化模型和优化器
    model = StructuralDamageNet(
        hidden=config.HIDDEN_DIM,
        dropout=config.DROPOUT_RATE,
        use_attention=use_attention,
        use_residual=use_residual
    ).to(config.DEVICE)

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}, 可训练参数: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 学习率调度器
    try:
        # 尝试使用verbose参数（较新版本的PyTorch）
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose = True
        )
    except TypeError:

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        print("[INFO] 当前PyTorch版本不支持ReduceLROnPlateau的verbose参数")

 
    ce_loss = nn.CrossEntropyLoss()  
    sev_loss = nn.MSELoss()  

    # AMP 混合精度
    scaler = torch.cuda.amp.GradScaler(enabled=amp and DEVICE.type == 'cuda')

    # 训练状态变量
    start_epoch = 1
    best_metric = float('inf')
    trigger = 0  # 早停计数器
    old_lr = lr  # 记录旧学习率

    # 恢复训练
    if resume and os.path.isfile(MODEL_PATH):
        try:
            # 尝试使用weights_only=False加载模型
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            # 如果当前PyTorch版本不支持weights_only参数，则使用默认方式加载
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
        old_lr = optimizer.param_groups[0]['lr']
        print(f"[INFO] 从 epoch {start_epoch} 恢复训练, 最佳指标: {best_metric:.4f}, 学习率: {old_lr:.2e}")
    else:
        print("[INFO] 从零开始训练")

    # 初始化日志文件
    os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)
    log_header = ['Epoch', 'train_loss', 'val_loss', 'loc_acc', 'sev_mae', 'lr', 'time']
    if not (resume and os.path.isfile(LOG_FILE)):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(log_header)

    # 训练循环
    best_metrics = {'val_loss': float('inf'), 'loc_acc': 0.0, 'sev_mae': float('inf')}

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_loc_loss = 0.0
        train_sev_loss = 0.0
        train_samples = 0

        # 使用tqdm显示训练进度
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} 训练", leave=False)
        for batch_idx, batch in enumerate(train_bar):
            if padding_strategy == 'mask':
                d, s, span, width, loc, sev, key_padding_mask = batch
            else:
                d, s, span, width, loc, sev = batch
            # 数据转移到设备
            if padding_strategy == 'mask':
                d, s, span, width, loc, sev = [t.to(config.DEVICE, non_blocking=True) for t in (d, s, span, width, loc, sev)]
                key_padding_mask = key_padding_mask.to(config.DEVICE)
            else:
                d, s, span, width, loc, sev = [t.to(config.DEVICE, non_blocking=True) for t in (d, s, span, width, loc, sev)]
            batch_size = d.size(0)

            optimizer.zero_grad(set_to_none=True)

            # 前向与损失（可选 AMP）
            with torch.cuda.amp.autocast(enabled=amp and config.DEVICE.type == 'cuda'):
                if padding_strategy == 'mask':
                    loc_log, pred_sev = model(d, s, span, width, key_padding_mask=key_padding_mask)
                else:
                    loc_log, pred_sev = model(d, s, span, width)
                loc_loss = ce_loss(loc_log, loc)
                sev_loss_val = sev_loss(pred_sev, sev)
                loss = loc_loss + 2.0 * sev_loss_val

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # 累积损失
            train_loss += loss.item() * batch_size
            train_loc_loss += loc_loss.item() * batch_size
            train_sev_loss += sev_loss_val.item() * batch_size
            train_samples += batch_size


            if batch_idx % 10 == 0:  
                train_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'loc_loss': f"{loc_loss.item():.4f}",
                    'sev_loss': f"{sev_loss_val.item():.4f}"
                })

        # 计算平均训练损失
        train_loss /= train_samples
        train_loc_loss /= train_samples
        train_sev_loss /= train_samples

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_loc_loss = 0.0
        val_sev_loss = 0.0
        val_samples = 0
        loc_corr = 0  
        sev_abs = 0.0 

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:03d} 验证", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                if padding_strategy == 'mask':
                    d, s, span, width, loc, sev, key_padding_mask = batch
                    d, s, span, width, loc, sev = [t.to(config.DEVICE, non_blocking=True) for t in (d, s, span, width, loc, sev)]
                    key_padding_mask = key_padding_mask.to(config.DEVICE)
                else:
                    d, s, span, width, loc, sev = batch
                    d, s, span, width, loc, sev = [t.to(config.DEVICE, non_blocking=True) for t in (d, s, span, width, loc, sev)]
                batch_size = d.size(0)

                with torch.cuda.amp.autocast(enabled=amp and config.DEVICE.type == 'cuda'):
                    if padding_strategy == 'mask':
                        loc_log, pred_sev = model(d, s, span, width, key_padding_mask=key_padding_mask)
                    else:
                        loc_log, pred_sev = model(d, s, span, width)

                # 计算损失
                loc_loss = ce_loss(loc_log, loc)
                sev_loss_val = sev_loss(pred_sev, sev)
                loss = loc_loss + 2.0 * sev_loss_val

                # 累积损失和指标
                val_loss += loss.item() * batch_size
                val_loc_loss += loc_loss.item() * batch_size
                val_sev_loss += sev_loss_val.item() * batch_size
                val_samples += batch_size
                pred_loc = loc_log.argmax(1)
                loc_corr += (pred_loc == loc).sum().item()
                sev_abs += (pred_sev - sev).abs().sum().item()


        val_loss /= val_samples
        val_loc_loss /= val_samples
        val_sev_loss /= val_samples
        loc_acc = loc_corr / val_samples 
        sev_mae = sev_abs / val_samples  # 损伤程度平均绝对误差
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']  # 当前学习率


        if val_loss < best_metrics['val_loss']:
            best_metrics['val_loss'] = val_loss
        if loc_acc > best_metrics['loc_acc']:
            best_metrics['loc_acc'] = loc_acc
        if sev_mae < best_metrics['sev_mae']:
            best_metrics['sev_mae'] = sev_mae

        # 更新学习率
        scheduler.step(val_loss)

        # 检查学习率是否变化
        if current_lr != old_lr:
            print(f"学习率从 {old_lr:.2e} 降低到 {current_lr:.2e}")
            old_lr = current_lr

        # 打印epoch结果
        epoch_msg = (f"[Epoch {epoch:03d}] "
                    f"train_loss={train_loss:.4f}(loc:{train_loc_loss:.4f}, sev:{train_sev_loss:.4f}) "
                    f"val_loss={val_loss:.4f}(loc:{val_loc_loss:.4f}, sev:{val_sev_loss:.4f}) "
                    f"loc_acc={loc_acc:.3f} sev_mae={sev_mae:.3f} "
                    f"lr={current_lr:.2e} time={epoch_time:.1f}s")
        print(epoch_msg)
        logger.info(epoch_msg)

        # 记录日志
        with open(config.log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, loc_acc, sev_mae, current_lr, epoch_time])

        # 确定当前指标值
        if config.BEST_METRIC == "val_loss":
            current_metric = val_loss
        elif config.BEST_METRIC == "loc_acc":
            current_metric = -loc_acc  # 负值因为我们要最小化指标
        elif config.BEST_METRIC == "sev_mae":
            current_metric = sev_mae

        # 保存最佳模型
        if current_metric < best_metric:
            best_metric = current_metric
            trigger = 0  # 重置早停计数器

            # 保存模型检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'sev_mean': ds.sev_mean,  # 保存标准化参数
                'sev_std': ds.sev_std,
                'd_mean': ds.d_mean,
                'd_std': ds.d_std,
                's_mean': ds.s_mean,
                's_std': ds.s_std,
                'span_mean': ds.span_mean,
                'span_std': ds.span_std,
                'width_mean': ds.width_mean,
                'width_std': ds.width_std,
                'config': {
                    'hidden_dim': config.HIDDEN_DIM,
                    'dropout_rate': config.DROPOUT_RATE,
                    'use_attention': use_attention,
                    'use_residual': use_residual
                }
            }

            torch.save(checkpoint, config.model_path)
            logger.info(f"保存最佳模型，{config.BEST_METRIC}: {best_metric:.4f}")
            print(f"保存最佳模型，{config.BEST_METRIC}: {best_metric:.4f}")
        else:
            trigger += 1
            if trigger >= patience:  # 早停检查
                print("早停触发")
                break

        # 定期保存检查点
        if epoch % save_every == 0:
            checkpoint_path = os.path.join(MODEL_DIR, f"damage_model_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'sev_mean': ds.sev_mean,
                'sev_std': ds.sev_std,
                'd_mean': ds.d_mean,
                'd_std': ds.d_std,
                's_mean': ds.s_mean,
                's_std': ds.s_std,
                'span_mean': ds.span_mean,
                'span_std': ds.span_std,
                'width_mean': ds.width_mean,
                'width_std': ds.width_std,
            }, checkpoint_path)

    logger.info(f"训练完成，最佳模型保存至 {config.model_path}")
    logger.info(f"最佳指标: {best_metrics}")
    print(f"训练完成，最佳模型保存至 {config.model_path}")
    print(f"最佳指标: {best_metrics}")

    return best_metrics


# 推理函数
@torch.no_grad()
def infer_csv(csv_path: str, output_format: str = "json") -> None:
    """对CSV文件进行推理
    功能亮点：
    1. 支持批量推理提高效率
    2. 自动反标准化输出结果
    3. 提供预测置信度
    参数：
    csv_path: 待推理数据CSV文件路径
    output_format: 输出格式（json或csv）
    """
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")

    # 加载模型
    try:
        # 尝试使用weights_only=False加载模型
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        # 如果当前PyTorch版本不支持weights_only参数，则使用默认方式加载
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = StructuralDamageNet().eval().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 获取标准化参数
    sev_mean = checkpoint.get('sev_mean', 0)
    sev_std = checkpoint.get('sev_std', 1)
    d_mean = checkpoint.get('d_mean', 0)
    d_std = checkpoint.get('d_std', 1)
    s_mean = checkpoint.get('s_mean', 0)
    s_std = checkpoint.get('s_std', 1)

    print(f"加载模型 (训练轮次: {checkpoint['epoch']})")
    print(f"严重程度标准化参数: 均值={sev_mean:.4f}, 标准差={sev_std:.4f}")
    print(f"距离标准化参数: 均值={d_mean:.4f}, 标准差={d_std:.4f}")
    print(f"斜率标准化参数: 均值={s_mean:.4f}, 标准差={s_std:.4f}")

    # 加载数据 - 测试时斜率放大倍数调整为合理值
    ds = BuildingDamageDataset(csv_path, has_labels=False, normalize=True,
                               scale_slope=1.0, is_test=True)
    BATCH_SIZE = 16
    loader = DataLoader(ds, BATCH_SIZE, shuffle=False, collate_fn=create_collate_fn('mask'))

    results = []  # 存储推理结果

    # 批量推理
    print(f"[DEBUG] 开始推理，共 {len(ds)} 个样本")
    print(f"[DEBUG] 标准化参数: sev_mean={sev_mean:.4f}, sev_std={sev_std:.4f}")

    for batch_idx, (d, s, span, width, _, _, key_padding_mask) in enumerate(loader):
        d, s, span, width = [t.to(DEVICE) for t in (d, s, span, width)]
        key_padding_mask = key_padding_mask.to(DEVICE)
        loc_log, pred_sev = model(d, s, span, width, key_padding_mask=key_padding_mask)
        loc_prob = torch.softmax(loc_log, dim=1)  # 计算位置预测置信度

        # 处理每个样本的预测结果
        for i in range(len(d)):
            # 反标准化严重程度预测
            pred_sev_denorm = pred_sev[i].item() * sev_std + sev_mean

            # 调试信息（前5个样本）
            if batch_idx == 0 and i < 5:
                print(f"[DEBUG] 样本 {i}: 原始预测={pred_sev[i].item():.4f}, "
                      f"反标准化后={pred_sev_denorm:.4f}")

            results.append({
                "pred_loc": int(loc_log[i].argmax()),  # 预测位置
                "pred_severity": float(pred_sev_denorm),  # 预测损伤程度
                "loc_conf": float(loc_prob[i].max())  # 位置预测置信度
            })

    # 保存结果
    if output_format == "json":
        out_path = csv_path.replace(".csv", "_pred.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("推理完成，结果保存至", out_path)
    else:  # CSV格式
        out_path = csv_path.replace(".csv", "_pred.csv")
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["pred_loc", "pred_severity", "loc_conf"])
            for res in results:
                writer.writerow([res["pred_loc"], res["pred_severity"], res["loc_conf"]])
        print("推理完成，结果保存至", out_path)

    return results


# 可视化工具
def plot_training_log(log_file: str = LOG_FILE):
    """绘制训练日志曲线
    功能：
    1. 绘制训练和验证损失曲线
    2. 绘制定位准确率曲线
    3. 绘制损伤程度MAE曲线
    4. 绘制学习率变化曲线
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("请安装matplotlib以使用可视化功能: pip install matplotlib")
        return
    if not os.path.isfile(log_file):
        print(f"日志文件未找到: {log_file}")
        return
    df = pd.read_csv(log_file)
    if len(df) == 0:
        print("日志文件为空")
        return
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # 损失曲线
    axes[0, 0].plot(df['Epoch'], df['train_loss'], label='训练损失')
    axes[0, 0].plot(df['Epoch'], df['val_loss'], label='验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # 定位准确率
    axes[0, 1].plot(df['Epoch'], df['loc_acc'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('定位准确率')
    axes[0, 1].grid(True)
    # 严重程度MAE
    axes[1, 0].plot(df['Epoch'], df['sev_mae'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('严重程度平均绝对误差')
    axes[1, 0].grid(True)
    # 学习率
    axes[1, 1].plot(df['Epoch'], df['lr'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('学习率变化')
    axes[1, 1].set_yscale('log')  # 对数刻度
    axes[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("训练曲线已保存至 training_curves.png")


# 注意力权重可视化工具
def visualize_attention(csv_path: str, sample_index: int = 0, layer_title: str = "Self-Attention"):
    """对指定CSV中的一个样本导出并可视化注意力权重。
    要求：使用padding_strategy='mask'以确保mask生效。
    """
    if not os.path.isfile(MODEL_PATH):
        print(f"模型文件未找到: {MODEL_PATH}")
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = StructuralDamageNet().eval().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    ds = BuildingDamageDataset(csv_path, has_labels=False, normalize=True, is_test=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=create_collate_fn('mask'))

    with torch.no_grad():
        for idx, (d, s, span, width, _, _, key_padding_mask) in enumerate(loader):
            if idx != sample_index:
                continue
            d, s, span, width = [t.to(DEVICE) for t in (d, s, span, width)]
            key_padding_mask = key_padding_mask.to(DEVICE)
            _, _, attn_weights = model(d, s, span, width, key_padding_mask=key_padding_mask, return_attention=True)
            # attn_weights: [B, num_heads, L, L] 或 [n_heads, B, L, L] 取决于实现；nn.MultiheadAttention返回的是 [B, L, L] 的平均权重（当average_attn_weights=True）。
            # 在PyTorch默认情况下，average_attn_weights=True，此时形状为 [B, L, L]。
            if attn_weights.dim() == 3:
                attn_map = attn_weights[0].detach().cpu().numpy()
            else:
                # 将多头取平均
                attn_map = attn_weights.mean(dim=1)[0].detach().cpu().numpy()

            # 根据mask裁剪到有效长度
            mask = key_padding_mask[0].detach().cpu().numpy()  # [L]
            valid_len = int((~mask).sum())
            attn_map = attn_map[:valid_len, :valid_len]

            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 5))
                plt.imshow(attn_map, cmap='viridis', aspect='auto')
                plt.colorbar(label='Attention')
                plt.title(f'{layer_title} (sample {sample_index})')
                plt.xlabel('Key positions')
                plt.ylabel('Query positions')
                plt.tight_layout()
                out_png = f'attention_sample{sample_index}.png'
                plt.savefig(out_png, dpi=300)
                plt.show()
                print(f"注意力热力图已保存至 {out_png}")
            except ImportError:
                print("请安装matplotlib以使用注意力可视化功能")
            break


# 严重程度预测优化函数
def optimize_severity_prediction(csv_path: str, output_path: str = None):
    """专门优化严重程度预测的函数
    功能：
    1. 使用更复杂的网络结构
    2. 调整损失函数权重
    3. 提供详细的预测分析
    """
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")

    # 加载模型
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = StructuralDamageNet().eval().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 获取标准化参数
    sev_mean = checkpoint.get('sev_mean', 0)
    sev_std = checkpoint.get('sev_std', 1)

    print(f"[优化] 加载模型 (训练轮次: {checkpoint['epoch']})")
    print(f"[优化] 严重程度标准化参数: 均值={sev_mean:.4f}, 标准差={sev_std:.4f}")

    # 加载数据
    ds = BuildingDamageDataset(csv_path, has_labels=False, normalize=True,
                               scale_slope=1.0, is_test=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=create_collate_fn('mask'))

    results = []
    severity_predictions = []

    print(f"[优化] 开始推理，共 {len(ds)} 个样本")

    for batch_idx, (d, s, span, width, _, _, key_padding_mask) in enumerate(loader):
        d, s, span, width = [t.to(DEVICE) for t in (d, s, span, width)]
        key_padding_mask = key_padding_mask.to(DEVICE)
        loc_log, pred_sev = model(d, s, span, width, key_padding_mask=key_padding_mask)
        loc_prob = torch.softmax(loc_log, dim=1)

        for i in range(len(d)):
            pred_sev_denorm = pred_sev[i].item() * sev_std + sev_mean
            severity_predictions.append(pred_sev_denorm)

            results.append({
                "pred_loc": int(loc_log[i].argmax()),
                "pred_severity": float(pred_sev_denorm),
                "loc_conf": float(loc_prob[i].max())
            })

    # 分析预测结果
    severity_predictions = np.array(severity_predictions)
    print(f"[优化] 严重程度预测统计:")
    print(f"  最小值: {severity_predictions.min():.4f}")
    print(f"  最大值: {severity_predictions.max():.4f}")
    print(f"  平均值: {severity_predictions.mean():.4f}")
    print(f"  标准差: {severity_predictions.std():.4f}")
    print(f"  中位数: {np.median(severity_predictions):.4f}")
    # 保存结果
    if output_path is None:
        output_path = csv_path.replace(".csv", "_optimized_pred.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[优化] 推理完成，结果保存至 {output_path}")
    return results


# 模型评估函数
@torch.no_grad()
def evaluate_model(csv_path: str):
    """评估模型性能
    功能：
    1. 计算定位准确率
    2. 计算损伤程度预测的MAE、RMSE和相关系数
    3. 绘制预测值与真实值的散点图
    """
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")

    # 加载模型
    try:
        # 尝试使用weights_only=False加载模型
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        # 如果当前PyTorch版本不支持weights_only参数，则使用默认方式加载
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = StructuralDamageNet().eval().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 获取标准化参数
    sev_mean = checkpoint.get('sev_mean', 0)
    sev_std = checkpoint.get('sev_std', 1)
    d_mean = checkpoint.get('d_mean', 0)
    d_std = checkpoint.get('d_std', 1)
    s_mean = checkpoint.get('s_mean', 0)
    s_std = checkpoint.get('s_std', 1)

    print(f"加载模型 (训练轮次: {checkpoint['epoch']})")
    print(f"严重程度标准化参数: 均值={sev_mean:.4f}, 标准差={sev_std:.4f}")
    print(f"距离标准化参数: 均值={d_mean:.4f}, 标准差={d_std:.4f}")
    print(f"斜率标准化参数: 均值={s_mean:.4f}, 标准差={s_std:.4f}")

    # 加载数据
    ds = BuildingDamageDataset(csv_path, has_labels=True, normalize=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=create_collate_fn('mask'))
    # 存储预测结果和真实值
    all_pred_locs = []
    all_true_locs = []
    all_pred_sevs = []
    all_true_sevs = []

    # 批量推理
    for d, s, span, width, loc, sev, key_padding_mask in loader:
        d, s, span, width, loc, sev = [t.to(DEVICE) for t in (d, s, span, width, loc, sev)]
        key_padding_mask = key_padding_mask.to(DEVICE)
        loc_log, pred_sev = model(d, s, span, width, key_padding_mask=key_padding_mask)
        # 反标准化严重程度预测
        pred_sev_denorm = pred_sev.cpu().numpy() * sev_std + sev_mean
        true_sev_denorm = sev.cpu().numpy() * sev_std + sev_mean
        # 收集预测结果和真实值
        all_pred_locs.extend(loc_log.argmax(1).cpu().numpy())
        all_true_locs.extend(loc.cpu().numpy())
        all_pred_sevs.extend(pred_sev_denorm)
        all_true_sevs.extend(true_sev_denorm)
    # 计算定位准确率
    loc_accuracy = np.mean(np.array(all_pred_locs) == np.array(all_true_locs))
    print(f"定位准确率: {loc_accuracy:.4f}")
    # 计算严重程度预测指标
    sev_mae = np.mean(np.abs(np.array(all_pred_sevs) - np.array(all_true_sevs)))
    sev_rmse = np.sqrt(np.mean((np.array(all_pred_sevs) - np.array(all_true_sevs)) ** 2))
    sev_corr = np.corrcoef(all_pred_sevs, all_true_sevs)[0, 1]
    print(f"严重程度预测:")
    print(f"  MAE: {sev_mae:.4f}")
    print(f"  RMSE: {sev_rmse:.4f}")
    print(f"  相关系数: {sev_corr:.4f}")
    # 绘制预测值与真实值的散点图
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.scatter(all_true_sevs, all_pred_sevs, alpha=0.5)
        # 添加理想预测线
        min_val = min(min(all_true_sevs), min(all_pred_sevs))
        max_val = max(max(all_true_sevs), max(all_pred_sevs))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('真实严重程度')
        plt.ylabel('预测严重程度')
        plt.title(f'严重程度预测 (MAE={sev_mae:.4f}, RMSE={sev_rmse:.4f})')
        plt.grid(True)
        plt.savefig('severity_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("严重程度预测散点图已保存至 severity_prediction.png")
    except ImportError:
        print("请安装matplotlib以使用可视化功能")
    return {
        'loc_accuracy': loc_accuracy,
        'sev_mae': sev_mae,
        'sev_rmse': sev_rmse,
        'sev_corr': sev_corr
    }


# 命令行接口
def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="建筑损伤诊断系统 - 专业版")
    sub = parser.add_subparsers(dest="cmd", required=True)
    # 训练命令
    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--csv", required=True, help="训练数据CSV文件路径")
    p_train.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    p_train.add_argument("--only_new", action="store_true", help="使用全部数据训练（无验证集）")
    p_train.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮次")
    p_train.add_argument("--lr", type=float, default=LR, help="学习率")
    p_train.add_argument("--wd", type=float, default=WD, help="权重衰减")
    p_train.add_argument("--patience", type=int, default=PATIENCE, help="早停耐心值")
    p_train.add_argument("--batch_size", type=int, help="训练批大小（默认自动推断，建议在GPU上调大）")
    p_train.add_argument("--num_workers", type=int, default=0, help="DataLoader工作进程数")
    p_train.add_argument("--amp", action="store_true", help="启用混合精度训练（需GPU）")
    p_train.add_argument("--benchmark", action="store_true", help="开启cudnn.benchmark提升卷积性能")
    # 推理命令
    p_infer = sub.add_parser("infer", help="对数据进行推理")
    p_infer.add_argument("--csv", required=True, help="待推理数据CSV文件路径")
    p_infer.add_argument("--format", choices=["json", "csv"], default="json", help="输出格式")
    # 可视化命令
    p_plot = sub.add_parser("plot", help="可视化训练曲线")
    p_plot.add_argument("--log", default=LOG_FILE, help="训练日志文件路径")
    # 注意力可视化命令
    p_attn = sub.add_parser("attn", help="可视化注意力权重热力图")
    p_attn.add_argument("--csv", required=True, help="待可视化的CSV文件路径（无标签也可）")
    p_attn.add_argument("--idx", type=int, default=0, help="样本索引（默认0）")
    # 评估命令
    p_eval = sub.add_parser("eval", help="评估模型性能")
    p_eval.add_argument("--csv", required=True, help="评估数据CSV文件路径")
    # 严重程度预测优化命令
    p_optimize = sub.add_parser("optimize", help="优化严重程度预测")
    p_optimize.add_argument("--csv", required=True, help="待推理数据CSV文件路径")
    p_optimize.add_argument("--output", help="输出文件路径")
    # 信息命令
    p_info = sub.add_parser("info", help="显示系统信息")
    args = parser.parse_args()
    # 根据命令执行相应操作
    if args.cmd == "train":
        train_model(
            args.csv,
            resume=args.resume,
            only_new=args.only_new,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
            benchmark=args.benchmark
        )
    elif args.cmd == "infer":
        infer_csv(args.csv, args.format)
    elif args.cmd == "plot":
        plot_training_log(args.log)
    elif args.cmd == "attn":
        visualize_attention(args.csv, args.idx)
    elif args.cmd == "eval":
        evaluate_model(args.csv)
    elif args.cmd == "optimize":
        optimize_severity_prediction(args.csv, args.output)
    elif args.cmd == "info":
        print("=" * 50)
        print("建筑物损伤诊断系统")
        print("")
        print("功能特点:")
        print("- 基于深度学习的建筑物损伤诊断")
        print("- 支持多任务学习（位置和程度预测）")
        print("- 自适应特征提取适用于不同尺寸建筑物")
        print("- 提供完整的训练、评估和可视化工具")
        print("=" * 50)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序运行时发生错误：{e}")
