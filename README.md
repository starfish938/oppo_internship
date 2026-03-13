# 基于深度学习的5G NR信道估计实习项目

## 项目简介
6周实习项目：基于深度学习的5G NR信道估计入门与实践。实现LS、MMSE、CNN三种信道估计算法。

## 环境配置
```bash
# 创建虚拟环境
conda create -n 5g_intern python=3.9 -y
conda activate 5g_intern

# 安装依赖
pip install numpy scipy matplotlib pandas tqdm torch torchvision jupyter
```

## 项目结构
```
5g_intern/
├── week1/          # 信号处理基础
├── week2/          # OFDM系统搭建
├── week3/          # LS信道估计
├── week4/          # 深度学习入门
├── week5/          # CNN信道估计
├── week6/          # 成果整理
├── utils/          # 工具函数
└── results/        # 实验结果
```



## 每周任务
- 第1周：频谱分析仪
- 第2周：OFDM仿真器
- 第3周：LS信道估计
- 第4周：MLP信道估计
- 第5周：CNN信道估计
- 第6周：技术报告
