import torch
import torch.nn as nn

class ChannelNet(nn.Module):
    """
    CNN信道估计网络 (ChannelNet, 参考 Ye et al. 2018)
    输入: [Batch, 2, N_subcarriers] (实部+虚部双通道)
    输出: [Batch, 2, N_subcarriers]
    """
    def __init__(self, n_subcarriers=52, n_filters=64):
        super(ChannelNet, self).__init__()
        self.conv1 = nn.Conv1d(2, n_filters, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(n_filters, n_filters, kernel_size=9, padding=4)
        self.conv4 = nn.Conv1d(n_filters, 2, kernel_size=9, padding=4)
        
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.bn3 = nn.BatchNorm1d(n_filters)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        residual = out
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out))) + residual
        out = self.conv4(out)
        return out