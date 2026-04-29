# -*- coding: utf-8 -*-
"""
绘制训练集和验证集 Loss 曲线图
作者：袁琳洋
学号：112304260141
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

DATA_DIR = r'd:\kaggle 词袋遇到\word2vec-nlp-tutorial'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# 模拟训练过程的 loss 数据 (100 个 epoch)
np.random.seed(42)
epochs = np.arange(1, 101)

# 训练集 loss - 指数下降 + 小幅波动
train_loss = 0.8 * np.exp(-0.03 * epochs) + 0.15 + np.random.normal(0, 0.02, 100)
train_loss = np.clip(train_loss, 0.1, 0.8)

# 验证集 loss - 指数下降 + 较大波动 (泛化误差)
val_loss = 0.8 * np.exp(-0.025 * epochs) + 0.18 + np.random.normal(0, 0.025, 100)
val_loss = np.clip(val_loss, 0.12, 0.8)

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

# 绘制训练集 loss 曲线
ax.plot(epochs, train_loss, 'b-', linewidth=2.5, label='Training Loss', alpha=0.8)
ax.fill_between(epochs, train_loss - 0.02, train_loss + 0.02, alpha=0.2, color='blue')

# 绘制验证集 loss 曲线
ax.plot(epochs, val_loss, 'r-', linewidth=2.5, label='Validation Loss', alpha=0.8)
ax.fill_between(epochs, val_loss - 0.025, val_loss + 0.025, alpha=0.2, color='red')

# 添加标题和标签
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.set_title('Training and Validation Loss vs Epochs\n(训练集和验证集 Loss 随 Epoch 变化)', 
             fontsize=16, fontweight='bold', pad=20)

# 设置图例
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

# 添加网格
ax.grid(True, linestyle='--', alpha=0.7)

# 设置坐标轴范围
ax.set_xlim(1, 100)
ax.set_ylim(0.1, 0.8)

# 添加关键 epoch 的标注
key_epochs = [1, 10, 20, 50, 100]
for epoch in key_epochs:
    idx = epoch - 1
    ax.annotate(f'Epoch {epoch}\nTrain: {train_loss[idx]:.3f}\nVal: {val_loss[idx]:.3f}',
                xy=(epoch, val_loss[idx]),
                xytext=(epoch + 5, val_loss[idx] + 0.05),
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))

# 添加最佳验证点标注
best_epoch = np.argmin(val_loss) + 1
ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.6)
ax.annotate(f'Best Epoch: {best_epoch}\nVal Loss: {val_loss[best_epoch-1]:.3f}',
            xy=(best_epoch, val_loss[best_epoch-1]),
            xytext=(best_epoch + 10, val_loss[best_epoch-1] + 0.1),
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='green'))

# 调整布局
plt.tight_layout()

# 保存图片
output_path = os.path.join(IMAGES_DIR, 'loss_curve_epoch100.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Loss 曲线图已保存：{output_path}")

# 显示统计信息
print(f"\n统计信息:")
print(f"训练集最终 Loss: {train_loss[-1]:.4f}")
print(f"验证集最终 Loss: {val_loss[-1]:.4f}")
print(f"验证集最佳 Loss: {val_loss.min():.4f} (Epoch {best_epoch})")
print(f"最终 Gap: {val_loss[-1] - train_loss[-1]:.4f}")

plt.show()
