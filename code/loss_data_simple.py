# -*- coding: utf-8 -*-
"""
绘制训练集和验证集 Loss 曲线图 - 纯 Python 版本
作者：袁琳洋
学号：112304260141
"""

import random
import math

# 生成模拟的 loss 数据
random.seed(42)

epochs = list(range(1, 101))
train_loss = []
val_loss = []

# 生成训练集 loss (指数下降)
for epoch in epochs:
    base = 0.8 * math.exp(-0.03 * epoch) + 0.15
    noise = random.gauss(0, 0.02)
    loss = max(0.1, min(0.8, base + noise))
    train_loss.append(round(loss, 4))

# 生成验证集 loss (下降稍慢，波动更大)
for epoch in epochs:
    base = 0.8 * math.exp(-0.025 * epoch) + 0.18
    noise = random.gauss(0, 0.025)
    loss = max(0.12, min(0.8, base + noise))
    val_loss.append(round(loss, 4))

# 打印数据
print("Epoch,Train_Loss,Val_Loss")
for i in range(100):
    print(f"{epochs[i]},{train_loss[i]},{val_loss[i]}")

# 统计信息
print("\n=== 统计信息 ===")
print(f"Epoch 1: Train = {train_loss[0]:.4f}, Val = {val_loss[0]:.4f}")
print(f"Epoch 10: Train = {train_loss[9]:.4f}, Val = {val_loss[9]:.4f}")
print(f"Epoch 50: Train = {train_loss[49]:.4f}, Val = {val_loss[49]:.4f}")
print(f"Epoch 100: Train = {train_loss[-1]:.4f}, Val = {val_loss[-1]:.4f}")

best_val = min(val_loss)
best_epoch = val_loss.index(best_val) + 1
print(f"\n验证集最佳 Loss: {best_val:.4f} (Epoch {best_epoch})")
print(f"最终 Gap: {val_loss[-1] - train_loss[-1]:.4f}")
