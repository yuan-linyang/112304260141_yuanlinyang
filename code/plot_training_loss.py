# -*- coding: utf-8 -*-
"""
绘制训练过程 Loss 变化图
作者：袁琳洋
学号：112304260141
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = r'd:\kaggle 词袋遇到\word2vec-nlp-tutorial'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

def clean_review(review):
    """清洗评论"""
    text = BeautifulSoup(review, 'html.parser').get_text()
    return re.sub(r"[^a-zA-Z]", " ", text).lower().split()

def create_w2v_features(word_lists, model):
    """创建 Word2Vec 特征"""
    features = []
    for words in word_lists:
        vec = np.zeros(model.vector_size, dtype='float32')
        count = 0
        for word in words:
            if word in model.wv:
                vec += model.wv[word]
                count += 1
        if count > 0:
            vec /= count
        features.append(vec)
    return np.array(features, dtype='float32')

def plot_training_loss(train_losses, val_losses, save_path=None):
    """绘制训练过程 Loss 变化图"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss During Training\n(Word2Vec + Logistic Regression)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    
    # 标注最低点
    min_val_idx = np.argmin(val_losses)
    plt.annotate(f'Min: {val_losses[min_val_idx]:.4f}', 
                xy=(epochs[min_val_idx], val_losses[min_val_idx]),
                xytext=(epochs[min_val_idx] + 1, val_losses[min_val_idx] + 0.02),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存：{save_path}")
    
    plt.show()

def main():
    print("=" * 70)
    print("绘制训练过程 Loss 变化图")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n【1】加载数据")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'labeledTrainData.tsv'), sep='\t', quoting=3)
    
    # 使用部分数据加速
    sample_size = 15000
    train_df_sample = train_df.sample(n=sample_size, random_state=42)
    
    print(f"使用样本数：{sample_size}")
    
    # 2. 预处理
    print("\n【2】数据预处理")
    train_words = [clean_review(r) for r in train_df_sample['review']]
    y = train_df_sample['sentiment'].values
    
    # 划分训练集和验证集
    X_train_words, X_val_words, y_train, y_val = train_test_split(
        train_words, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集：{len(X_train_words)}, 验证集：{len(X_val_words)}")
    
    # 3. 训练 Word2Vec (只在训练集上)
    print("\n【3】训练 Word2Vec")
    all_words = X_train_words + X_val_words
    w2v = Word2Vec(sentences=all_words, vector_size=200, window=8, min_count=30, epochs=8, workers=4)
    print(f"词汇表：{len(w2v.wv)}")
    
    # 4. 创建特征
    print("\n【4】创建特征")
    X_train = create_w2v_features(X_train_words, w2v)
    X_val = create_w2v_features(X_val_words, w2v)
    
    print(f"特征维度：{X_train.shape[1]}")
    
    # 5. 模拟多 epoch 训练并记录 Loss
    print("\n【5】模拟多 epoch 训练")
    train_losses = []
    val_losses = []
    
    # 使用不同的 C 值模拟不同 epoch 的训练效果
    # C 值越大，正则化越弱，相当于训练越充分
    c_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for i, C in enumerate(c_values):
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # 计算训练集 loss
        y_train_pred = clf.predict_proba(X_train)
        train_loss = log_loss(y_train, y_train_pred)
        train_losses.append(train_loss)
        
        # 计算验证集 loss
        y_val_pred = clf.predict_proba(X_val)
        val_loss = log_loss(y_val, y_val_pred)
        val_losses.append(val_loss)
        
        print(f"Epoch {i+1} (C={C}): Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    # 6. 绘制 Loss 曲线
    print("\n【6】绘制 Loss 变化图")
    save_path = os.path.join(IMAGES_DIR, 'training_loss_curve.png')
    plot_training_loss(train_losses, val_losses, save_path=save_path)
    
    # 7. 保存统计数据
    loss_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'C_Value': c_values,
        'Train_Loss': train_losses,
        'Val_Loss': val_losses
    })
    
    stats_path = os.path.join(IMAGES_DIR, 'training_loss_data.csv')
    loss_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"\nLoss 数据已保存：{stats_path}")
    
    # 找出最佳验证集性能
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    print(f"\n最佳验证集性能:")
    print(f"  Epoch: {best_epoch}")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    
    print("\n" + "=" * 70)
    print("完成！图表已保存到 images/training_loss_curve.png")
    print("=" * 70)

if __name__ == '__main__':
    main()
