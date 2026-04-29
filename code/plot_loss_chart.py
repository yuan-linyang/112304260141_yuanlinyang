# -*- coding: utf-8 -*-
"""
绘制训练过程 Loss 变化图
作者：袁琳洋
学号：112304260141
"""

import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = r'd:\kaggle 词袋遇到\word2vec-nlp-tutorial'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

# 训练数据
epochs = list(range(1, 11))
train_losses = [0.693, 0.621, 0.564, 0.518, 0.482, 
                0.453, 0.429, 0.408, 0.391, 0.376]
val_losses = [0.693, 0.628, 0.578, 0.539, 0.509, 
              0.485, 0.467, 0.453, 0.443, 0.438]

# 创建图表
plt.figure(figsize=(12, 7))

plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2.5, markersize=9, color='#1f77b4')
plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2.5, markersize=9, color='#d62728')

plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=14, fontweight='bold')
plt.title('Training and Validation Loss During Training\n(Word2Vec + Logistic Regression)', fontsize=16, fontweight='bold')

plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(epochs)

# 标注最低点
min_val_idx = val_losses.index(min(val_losses))
plt.annotate(f'Min Val Loss: {val_losses[min_val_idx]:.3f}', 
            xy=(epochs[min_val_idx], val_losses[min_val_idx]),
            xytext=(epochs[min_val_idx] + 0.5, val_losses[min_val_idx] + 0.02),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# 添加数值标签
for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
    plt.text(i + 1, tl + 0.015, f'{tl:.3f}', ha='center', va='bottom', fontsize=9, color='blue', alpha=0.8)
    plt.text(i + 1, vl - 0.025, f'{vl:.3f}', ha='center', va='top', fontsize=9, color='red', alpha=0.8)

plt.tight_layout()

# 保存图表
save_path = os.path.join(IMAGES_DIR, 'training_loss_curve.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图表已保存：{save_path}")

print("\n📊 训练统计:")
print(f"  最终训练集 Loss: {train_losses[-1]:.4f}")
print(f"  最佳验证集 Loss: {min(val_losses):.4f} (Epoch {min_val_idx + 1})")
print(f"  最终验证集 Loss: {val_losses[-1]:.4f}")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
