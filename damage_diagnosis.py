"""
建筑物损伤诊断系统 - 专业版
Copyright (c) 2025 Chen Yanxuan. 保留所有权利。

本软件仅供评估和学习目的使用。未经明确书面许可，不得用于商业用途。
如果您对合作机会感兴趣，请联系: achenyanxuan@163.com

功能：
1. 从CSV文件读取有限元模拟数据和实际损伤案例进行训练和验证
2. 基于InSAR数据反演建筑物损伤情况

输入文件：
- train.csv：训练数据（有限元批量仿真的损伤情况+工况）
- val.csv：验证数据（独立于训练集的有限元模拟数据及实际损伤案例数据）
- test_building.csv：待推理数据（InSAR得到的实际建筑物屋面损伤情况）

输出文件：
1. damage_model.pt：训练后的模型权重
2. training_log.csv：训练日志，记录每个epoch的损失、准确率和误差
3. test_building_pred.json：推理结果，包含预测的象限编号、损伤类型、损伤程度和置信度

作者：Chen Yanxuan
联系方式：chenyanxuan@163.com
WX:CYX241（注明来意）
日期：2025-08-15
版本：1.0.0
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ---------- 超参数设置 ----------
N_QUADRANT = 3  # 象限数量
MODEL_DIR = "models"  # 模型保存目录
MODEL_PATH = os.path.join(MODEL_DIR, "damage_model.pt")  # 模型保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择（GPU/CPU）
SEED = 42  # 随机种子
EPOCHS = 1000  # 训练轮数
LR = 1e-3  # 学习率
WD = 1e-5  # 权重衰减
PATIENCE = 100  # 早停耐心值
SAVE_EVERY = 50  # 每隔多少轮保存一次检查点
LOG_FILE = "training_log.csv"  # 训练日志文件
BEST_METRIC = "val_loss"  # 最佳模型评估指标


def set_seed(seed: int = SEED):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ---------- 数据集类 ----------
class BuildingDamageDataset(Dataset):
    """建筑物损伤数据集类

    功能：
    1. 读取CSV文件并解析数据
    2. 对数据进行标准化处理
    3. 支持训练和测试模式

    参数：
    csv_path: CSV文件路径
    has_labels: 数据是否包含标签
    normalize: 是否进行标准化
    scale_slope: 斜率缩放因子
    is_test: 是否为测试模式
    """

    def __init__(self, csv_path: str, has_labels: bool = True, normalize: bool = True,
                 scale_slope: float = 1.0, is_test: bool = False):
        self.has_labels = has_labels
        self.normalize = normalize
        self.scale_slope = scale_slope
        self.is_test = is_test

        # 读取CSV文件
        rows = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            rdr = csv.reader(f)
            for line in rdr:
                if not line:  # 跳过空行
                    continue
                rows.append([float(v) for v in line])  # 转换为浮点数

        # 验证数据格式
        if has_labels and any(len(r) < 2 for r in rows):
            raise ValueError("CSV 至少需要 2 列 (severity, loc)")

        self.samples = []
        # 初始化标准化参数
        self.d_mean, self.d_std = 0, 1  # 距离的均值和标准差
        self.s_mean, self.s_std = 0, 1  # 斜率的均值和标准差
        self.sev_mean, self.sev_std = 0, 1  # 损伤程度的均值和标准差

        # 计算均值和标准差用于标准化
        if normalize and len(rows) > 0:
            all_d, all_s, all_sev = [], [], []  # 存储所有距离、斜率和损伤程度
            for row in rows:
                # 提取特征（最后一列是位置，倒数第二列是损伤程度）
                feats = row[:-2] if has_labels else row
                if len(feats) % 2 == 1:  # 确保特征数量是偶数
                    feats = feats[:-1]
                if len(feats) == 0:  # 跳过空特征
                    continue

                # 提取距离和斜率特征
                d_vec = feats[0::2]  # 距离特征（偶数索引）
                s_vec = feats[1::2]  # 斜率特征（奇数索引）

                # 测试模式下放大斜率
                if is_test:
                    s_vec = [s * scale_slope for s in s_vec]

                # 收集所有数据用于计算统计量
                all_d.extend(d_vec)
                all_s.extend(s_vec)

                if has_labels:
                    sev = float(row[-2])  # 损伤程度
                    all_sev.append(sev)

            # 计算距离和斜率的均值和标准差
            if all_d and all_s:
                self.d_mean, self.d_std = np.mean(all_d), np.std(all_d)
                self.s_mean, self.s_std = np.mean(all_s), np.std(all_s)
                # 避免除零错误
                self.d_std = self.d_std if self.d_std > 0 else 1
                self.s_std = self.s_std if self.s_std > 0 else 1

            # 计算损伤程度的均值和标准差
            if all_sev:
                self.sev_mean, self.sev_std = np.mean(all_sev), np.std(all_sev)
                self.sev_std = self.sev_std if self.sev_std > 0 else 1

        # 处理每一行数据
        for row in rows:
            if has_labels:
                sev = float(row[-2])  # 损伤程度
                loc = int(row[-1])  # 损伤位置（象限）
                feats = row[:-2]  # 特征
            else:
                feats = row
                sev, loc = 0.0, 0  # 测试数据使用默认值

            # 确保特征数量是偶数
            if len(feats) % 2 == 1:
                feats = feats[:-1]
            if len(feats) == 0:  # 跳过空特征
                continue

            # 提取距离和斜率特征
            d_vec = feats[0::2]
            s_vec = feats[1::2]

            # 测试模式下放大斜率
            if is_test:
                s_vec = [s * scale_slope for s in s_vec]

            # 标准化数据
            if normalize:
                d_vec = [(d - self.d_mean) / self.d_std for d in d_vec]
                s_vec = [(s - self.s_mean) / self.s_std for s in s_vec]
                if has_labels:
                    sev = (sev - self.sev_mean) / self.sev_std

            # 存储处理后的样本
            self.samples.append((
                torch.tensor(d_vec, dtype=torch.float32),  # 距离特征
                torch.tensor(s_vec, dtype=torch.float32),  # 斜率特征
                torch.tensor(loc, dtype=torch.long),  # 位置标签
                torch.tensor(sev, dtype=torch.float32)  # 损伤程度标签
            ))

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        return self.samples[idx]


def collate_fn(batch):
    """自定义批处理函数，用于处理可变长度序列

    功能：
    1. 对批次中的可变长度序列进行填充
    2. 返回填充后的张量

    参数：
    batch: 批数据

    返回：
    填充后的距离、斜率、位置和损伤程度张量
    """
    d, s, loc, sev = zip(*batch)
    # 对距离和斜率进行填充
    d_pad = pad_sequence(d, batch_first=True, padding_value=0.0).float()
    s_pad = pad_sequence(s, batch_first=True, padding_value=0.0).float()
    return d_pad, s_pad, torch.stack(loc), torch.stack(sev)


# ---------- 神经网络模型 ----------
class StructuralDamageNet(nn.Module):
    """建筑物结构损伤诊断神经网络

    创新特点：
    1. 双流架构同时处理距离和斜率特征
    2. 多任务学习同时预测损伤位置和程度
    3. 自适应特征提取适用于不同尺寸的建筑物

    参数：
    hidden: 隐藏层维度
    dropout: Dropout比率
    """

    def __init__(self, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        # 1D卷积网络
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),  # 输入通道2（距离和斜率），输出通道64
            nn.BatchNorm1d(64),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Dropout(dropout),  # Dropout防止过拟合
            nn.Conv1d(64, 128, kernel_size=5, padding=2),  # 第二层卷积
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),  # 自适应最大池化
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # 位置预测头（分类任务）
        self.loc_head = nn.Linear(hidden, N_QUADRANT)

        # 损伤程度预测头（回归任务）
        self.sev_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 4, 1)  # 输出一个标量值
        )

    def forward(self, d, s):
        """前向传播

        参数：
        d: 距离特征
        s: 斜率特征

        返回：
        loc_head: 位置预测（分类）
        sev_head: 损伤程度预测（回归）
        """
        # 将距离和斜率堆叠为2通道输入
        x = torch.stack([d, s], dim=1)
        # 通过卷积网络提取特征
        h = self.cnn(x).squeeze(-1)
        # 通过全连接层
        h = self.fc(h)
        # 返回位置预测和损伤程度预测
        return self.loc_head(h), self.sev_head(h).squeeze(-1)


# ---------- 训练函数 ----------
def train_model(csv_path: str,
                resume: bool = False,
                only_new: bool = False,
                epochs: int = EPOCHS,
                lr: float = LR,
                wd: float = WD,
                patience: int = PATIENCE,
                save_every: int = SAVE_EVERY) -> None:
    """训练模型

    功能亮点：
    1. 支持从检查点恢复训练
    2. 动态学习率调整
    3. 早停机制防止过拟合
    4. 多任务损失平衡

    参数：
    csv_path: 训练数据CSV文件路径
    resume: 是否从检查点恢复训练
    only_new: 是否使用全部数据训练（无验证集）
    epochs: 训练轮数
    lr: 学习率
    wd: 权重衰减
    patience: 早停耐心值
    save_every: 每隔多少轮保存一次检查点
    """
    set_seed()  # 设置随机种子

    # 创建模型目录
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 加载数据集
    ds = BuildingDamageDataset(csv_path, has_labels=True, normalize=True)
    # 动态计算批次大小
    BATCH_SIZE = min(64, max(2, len(ds) // 10 + 1))

    # 划分训练集和验证集
    if only_new or len(ds) < 10:
        train_ds = val_ds = ds
        print(f"[INFO] 使用全部 {len(ds)} 个样本进行训练和验证")
    else:
        split = int(0.8 * len(ds))
        train_ds, val_ds = random_split(ds, [split, len(ds) - split])
        print(f"[INFO] 数据集分割: {len(train_ds)} 训练, {len(val_ds)} 验证")

    # 创建数据加载器
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)

    # 初始化模型和优化器
    model = StructuralDamageNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 学习率调度器
    try:
        # 尝试使用verbose参数（较新版本的PyTorch）
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
    except TypeError:
        # 如果verbose参数不可用，使用旧版本的方式
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        print("[INFO] 当前PyTorch版本不支持ReduceLROnPlateau的verbose参数")

    # 损失函数
    ce_loss = nn.CrossEntropyLoss()  # 分类损失（位置预测）
    sev_loss = nn.HuberLoss(delta=1.0)  # 回归损失（损伤程度预测），对异常值更鲁棒

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
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0
        train_loc_loss = 0
        train_sev_loss = 0

        # 使用tqdm显示训练进度
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} 训练")
        for d, s, loc, sev in train_bar:
            # 数据转移到设备
            d, s, loc, sev = [t.to(DEVICE) for t in (d, s, loc, sev)]
            # 前向传播
            loc_log, pred_sev = model(d, s)

            # 计算损失
            loc_loss = ce_loss(loc_log, loc)  # 位置损失
            sev_loss_val = sev_loss(pred_sev, sev)  # 损伤程度损失

            # 加权组合损失 - 给严重程度预测更高的权重
            loss = loc_loss + 2.0 * sev_loss_val

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积损失
            train_loss += loss.item() * d.size(0)
            train_loc_loss += loc_loss.item() * d.size(0)
            train_sev_loss += sev_loss_val.item() * d.size(0)

            # 更新进度条显示
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'loc_loss': f"{loc_loss.item():.4f}",
                'sev_loss': f"{sev_loss_val.item():.4f}"
            })

        # 计算平均训练损失
        train_loss /= len(train_ds)
        train_loc_loss /= len(train_ds)
        train_sev_loss /= len(train_ds)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_loc_loss = 0
        val_sev_loss = 0
        n = 0
        loc_corr = 0  # 正确预测的位置数量
        sev_abs = 0  # 损伤程度绝对误差总和

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:03d} 验证")
        with torch.no_grad():
            for d, s, loc, sev in val_bar:
                d, s, loc, sev = [t.to(DEVICE) for t in (d, s, loc, sev)]
                loc_log, pred_sev = model(d, s)

                # 计算损失
                loc_loss = ce_loss(loc_log, loc)
                sev_loss_val = sev_loss(pred_sev, sev)
                loss = loc_loss + 2.0 * sev_loss_val

                # 累积损失和指标
                val_loss += loss.item() * d.size(0)
                val_loc_loss += loc_loss.item() * d.size(0)
                val_sev_loss += sev_loss_val.item() * d.size(0)
                n += d.size(0)
                pred_loc = loc_log.argmax(1)
                loc_corr += (pred_loc == loc).sum().item()
                sev_abs += (pred_sev - sev).abs().sum().item()

        # 计算平均验证指标
        val_loss /= n
        val_loc_loss /= n
        val_sev_loss /= n
        loc_acc = loc_corr / n  # 位置准确率
        sev_mae = sev_abs / n  # 损伤程度平均绝对误差
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']  # 当前学习率

        # 更新学习率
        scheduler.step(val_loss)

        # 检查学习率是否变化
        if current_lr != old_lr:
            print(f"学习率从 {old_lr:.2e} 降低到 {current_lr:.2e}")
            old_lr = current_lr

        # 打印epoch结果
        print(f"[Epoch {epoch:03d}] "
              f"train_loss={train_loss:.4f}(loc:{train_loc_loss:.4f}, sev:{train_sev_loss:.4f}) "
              f"val_loss={val_loss:.4f}(loc:{val_loc_loss:.4f}, sev:{val_sev_loss:.4f}) "
              f"loc_acc={loc_acc:.3f} sev_mae={sev_mae:.3f} "
              f"lr={current_lr:.2e} time={epoch_time:.1f}s")

        # 记录日志
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, loc_acc, sev_mae, current_lr, epoch_time])

        # 确定当前指标值
        if BEST_METRIC == "val_loss":
            current_metric = val_loss
        elif BEST_METRIC == "loc_acc":
            current_metric = -loc_acc  # 负值因为我们要最小化指标
        elif BEST_METRIC == "sev_mae":
            current_metric = sev_mae

        # 保存最佳模型
        if current_metric < best_metric:
            best_metric = current_metric
            trigger = 0  # 重置早停计数器
            torch.save({
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
            }, MODEL_PATH)
            print(f"保存最佳模型，{BEST_METRIC}: {best_metric:.4f}")
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
            }, checkpoint_path)

    print("训练完成，最佳模型保存至", MODEL_PATH)


# ---------- 推理函数 ----------
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

    # 加载数据 - 测试时斜率放大10000倍
    ds = BuildingDamageDataset(csv_path, has_labels=False, normalize=True,
                           scale_slope=10000.0, is_test=True)
    BATCH_SIZE = 64
    loader = DataLoader(ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    results = []  # 存储推理结果

    # 批量推理
    for d, s, _, _ in loader:
        d, s = [t.to(DEVICE) for t in (d, s)]
        loc_log, pred_sev = model(d, s)
        loc_prob = torch.softmax(loc_log, dim=1)  # 计算位置预测置信度

        # 处理每个样本的预测结果
        for i in range(len(d)):
            # 反标准化严重程度预测
            pred_sev_denorm = pred_sev[i].item() * sev_std + sev_mean

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


# ---------- 可视化工具 ----------
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


# ---------- 模型评估函数 ----------
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
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # 存储预测结果和真实值
    all_pred_locs = []
    all_true_locs = []
    all_pred_sevs = []
    all_true_sevs = []

    # 批量推理
    for d, s, loc, sev in loader:
        d, s, loc, sev = [t.to(DEVICE) for t in (d, s, loc, sev)]
        loc_log, pred_sev = model(d, s)

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
    sev_rmse = np.sqrt(np.mean((np.array(all_pred_sevs) - np.array(all_true_sevs))**2))
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


# ---------- 命令行接口 ----------
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

    # 推理命令
    p_infer = sub.add_parser("infer", help="对数据进行推理")
    p_infer.add_argument("--csv", required=True, help="待推理数据CSV文件路径")
    p_infer.add_argument("--format", choices=["json", "csv"], default="json", help="输出格式")

    # 可视化命令
    p_plot = sub.add_parser("plot", help="可视化训练曲线")
    p_plot.add_argument("--log", default=LOG_FILE, help="训练日志文件路径")

    # 评估命令
    p_eval = sub.add_parser("eval", help="评估模型性能")
    p_eval.add_argument("--csv", required=True, help="评估数据CSV文件路径")

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
            patience=args.patience
        )
    elif args.cmd == "infer":
        infer_csv(args.csv, args.format)
    elif args.cmd == "plot":
        plot_training_log(args.log)
    elif args.cmd == "eval":
        evaluate_model(args.csv)
    elif args.cmd == "info":
        print("=" * 50)
        print("建筑物损伤诊断系统 - 专业版")
        print("=" * 50)
        print("作者: Chen Yanxuan")
        print("联系方式: chenyanxuan@example.com")
        print("GitHub: https://github.com/yourusername/building-damage-diagnosis")
        print("版本: 1.0.0")
        print("发布日期: 2025-08-15")
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
    main()