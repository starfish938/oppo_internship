# 第一周学习报告

## Day 1：信号与频谱
- 学习了正弦波、方波、chirp信号的生成
- 掌握了FFT频谱分析方法
- 实现了时域波形、幅度谱、相位谱的可视化

## Day 2：数字调制
- 实现了QPSK调制解调（2比特→1符号）
- 实现了16QAM调制解调（4比特→1符号）
- 绘制了星座图，理解符号映射关系

## Day 3：AWGN信道与BER分析
- 实现了AWGN信道模型
- 完成了BPSK/QPSK/16QAM在不同SNR下的BER仿真
- 对比了仿真曲线与理论曲线
- <img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/ee30c560-3c39-4594-ad9f-69d9b1026429" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/35754582-40f6-4b5d-aa61-cb23366be483" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/7e9a21dd-30f3-468e-b356-ab202f16a7dc" />



## Day 4：代码封装
- 将前3天代码整理成工具库 `signal_utils.py`
- 封装了5个核心函数：
  - `fft_analysis()`：频谱分析
  - `modulate()`：通用调制
  - `demodulate()`：通用解调
  - `add_awgn()`：添加噪声
  - `calculate_ber()`：计算误码率

## Day 5：频谱分析仪项目
- 用GridSpec创建三视图布局
- 实现了时域波形、频谱图、STFT时频图
- 支持矩形窗和汉明窗切换
- 测试了正弦波、方波、chirp、噪声四种信号

## 收获与总结
- 掌握了信号处理基础：FFT、窗函数、STFT
- 理解了数字调制原理：QPSK、16QAM
- 学会了通信系统仿真流程：调制→信道→解调→BER分析
- 完成了第一个综合项目：频谱分析仪
