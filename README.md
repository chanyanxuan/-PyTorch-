<img width="3353" height="1874" alt="38210fa8-43aa-49b3-98a6-dcbef30ee32b" src="https://github.com/user-attachments/assets/38fd8f2b-5b9d-4a11-ae3f-6aa7478a4d3c" /># 建筑物损伤诊断系统

基于深度学习的建筑物损伤诊断系统，使用 PyTorch 实现，能够从 InSAR 数据中识别和定位建筑物损伤。
<img width="3353" height="1874" alt="38210fa8-43aa-49b3-98a6-dcbef30ee32b" src="https://github.com/user-attachments/assets/412f929d-e4ea-486b-985f-119e2b131d27" />

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

# 功能特点

- **多任务学习**: 同时预测损伤位置（象限）和损伤程度
- **自适应架构**: 适用于不同尺寸和形状的建筑物
- **完整工具链**: 训练、评估、推理和可视化一体化
- **高性能**: 支持 GPU 加速和批量处理
- **易用性**: 简单的命令行接口和清晰的文档

#  安装

# 前提条件
- Python 3.8+
- PyTorch 1.9+

# 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/building-damage-diagnosis.git
cd building-damage-diagnosis

# 模型架构
系统使用基于 CNN 的多任务学习架构：
特征提取: 1D 卷积网络处理距离和斜率特征

多任务头:
分类头：预测损伤位置（象限）
回归头：预测损伤程度

 性能指标
在测试数据上的典型性能：

位置准确率: >85%

损伤程度 MAE: <0.15

推理速度: 1000+ samples/second (GPU)

 开发者
Chen Yanxuan

 贡献
欢迎提交 Issue 和 Pull Request！对于重大更改，请先开 Issue 讨论您想要更改的内容。

 许可证
本项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情。

致谢
感谢 PyTorch 团队提供的深度学习框架

感谢所有为这个项目提供反馈和建议的贡献者

⭐ 如果这个项目对你有帮助，请给它一个 Star！
