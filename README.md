# 🏢 建筑物损伤诊断系统

基于深度学习的建筑物损伤诊断系统，使用 PyTorch 实现，能够从 InSAR 数据中识别和定位建筑物损伤。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ 功能特点

- **多任务学习**: 同时预测损伤位置（象限）和损伤程度
- **自适应架构**: 适用于不同尺寸和形状的建筑物
- **完整工具链**: 训练、评估、推理和可视化一体化
- **高性能**: 支持 GPU 加速和批量处理
- **易用性**: 简单的命令行接口和清晰的文档

## 📦 安装

### 前提条件
- Python 3.8+
- PyTorch 1.9+

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/building-damage-diagnosis.git
cd building-damage-diagnosis
