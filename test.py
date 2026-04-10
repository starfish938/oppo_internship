# test_imports.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("测试导入...")

try:
    from models.ls_estimator import ls_channel_estimation, linear_interpolation
    print("✓ ls_estimator 导入成功")
except Exception as e:
    print(f"✗ ls_estimator 导入失败: {e}")

try:
    from models.mmse_estimator import mmse_channel_estimation
    print("✓ mmse_estimator 导入成功")
except Exception as e:
    print(f"✗ mmse_estimator 导入失败: {e}")

try:
    from models.cnn_estimator import ChannelNet
    print("✓ cnn_estimator 导入成功")
except Exception as e:
    print(f"✗ cnn_estimator 导入失败: {e}")

try:
    from utils.channel_model import generate_rayleigh_channel
    print("✓ channel_model 导入成功")
except Exception as e:
    print(f"✗ channel_model 导入失败: {e}")

try:
    from utils.metrics import calculate_nmse
    print("✓ metrics 导入成功")
except Exception as e:
    print(f"✗ metrics 导入失败: {e}")

print("测试完成！")