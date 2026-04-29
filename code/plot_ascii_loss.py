# -*- coding: utf-8 -*-
"""
使用 ASCII 字符绘制 Loss 曲线图
作者：袁琳洋
学号：112304260141
"""

import random
import math

# 生成数据
random.seed(42)
epochs = list(range(1, 101))
train_loss = []
val_loss = []

for epoch in epochs:
    base = 0.8 * math.exp(-0.03 * epoch) + 0.15
    noise = random.gauss(0, 0.02)
    train_loss.append(max(0.1, min(0.8, base + noise)))

for epoch in epochs:
    base = 0.8 * math.exp(-0.025 * epoch) + 0.18
    noise = random.gauss(0, 0.025)
    val_loss.append(max(0.12, min(0.8, base + noise)))

# 创建 ASCII 图表
width = 100
height = 30

print("=" * (width + 10))
print(" " * 30 + "Training and Validation Loss vs Epochs")
print(" " * 35 + "训练集和验证集 Loss 随 Epoch 变化")
print("=" * (width + 10))
print()

# 创建画布
canvas = [[' ' for _ in range(width)] for _ in range(height)]

# 找到最小和最大值
min_loss = 0.1
max_loss = 0.8

# 绘制训练集曲线 (使用 '*')
for i, epoch in enumerate(epochs):
    x = int((i / 99) * (width - 1))
    y = int(height - 1 - ((train_loss[i] - min_loss) / (max_loss - min_loss)) * (height - 1))
    if 0 <= y < height:
        canvas[y][x] = '*'

# 绘制验证集曲线 (使用 'o')
for i, epoch in enumerate(epochs):
    x = int((i / 99) * (width - 1))
    y = int(height - 1 - ((val_loss[i] - min_loss) / (max_loss - min_loss)) * (height - 1))
    if 0 <= y < height:
        if canvas[y][x] == '*':
            canvas[y][x] = 'X'  # 重叠点
        else:
            canvas[y][x] = 'o'

# 打印 Y 轴和图表
print("Loss")
print("0.80 |", end='')
for row in canvas[:height//3]:
    print(''.join(row))
    print("     |", end='')
print()

print("0.45 |", end='')
for row in canvas[height//3:2*height//3]:
    print(''.join(row))
    print("     |", end='')
print()

print("0.10 |", end='')
for row in canvas[2*height//3:]:
    print(''.join(row))
    print("     |", end='')
print()

# X 轴
print("     +" + "-" * width)
print("      1" + " " * 30 + "50" + " " * 30 + "100")
print(" " * 45 + "Epoch")
print()

# 图例
print("图例:")
print("  *  Training Loss (训练集)")
print("  o  Validation Loss (验证集)")
print("  X  重叠点")
print()

# 统计信息
print("=" * 60)
print("统计信息:")
print("=" * 60)
print(f"Epoch 1:   Train = {train_loss[0]:.4f}, Val = {val_loss[0]:.4f}")
print(f"Epoch 10:  Train = {train_loss[9]:.4f}, Val = {val_loss[9]:.4f}")
print(f"Epoch 50:  Train = {train_loss[49]:.4f}, Val = {val_loss[49]:.4f}")
print(f"Epoch 100: Train = {train_loss[-1]:.4f}, Val = {val_loss[-1]:.4f}")
print()

best_val = min(val_loss)
best_epoch = val_loss.index(best_val) + 1
print(f"验证集最佳 Loss: {best_val:.4f} (Epoch {best_epoch})")
print(f"最终 Gap: {val_loss[-1] - train_loss[-1]:.4f}")
print("=" * 60)
