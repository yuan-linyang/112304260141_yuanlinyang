# -*- coding: utf-8 -*-
"""
生成改进版提交文件 - 简化快速版
"""

import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec

DATA_DIR = r'd:\kaggle 词袋遇到\word2vec-nlp-tutorial'
SUBMISSION_DIR = os.path.join(DATA_DIR, 'submission')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def clean_review(review):
    text = BeautifulSoup(review, 'html.parser').get_text()
    return re.sub(r"[^a-zA-Z]", " ", text).lower().split()

print("=" * 70)
print("生成改进版提交文件 - 快速版")
print("=" * 70)

# 1. 加载数据
print("\n【1】加载数据")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'labeledTrainData.tsv'), sep='\t', quoting=3)
test_df = pd.read_csv(os.path.join(DATA_DIR, 'testData.tsv'), sep='\t', quoting=3)
unlabeled_df = pd.read_csv(os.path.join(DATA_DIR, 'unlabeledTrainData.tsv'), sep='\t', quoting=3)

# 2. 预处理
print("\n【2】预处理")
train_words = [clean_review(r) for r in train_df['review']]
test_words = [clean_review(r) for r in test_df['review']]
all_words = train_words + test_words

y_train = train_df['sentiment'].values
train_texts = [' '.join(w) for w in train_words]
test_texts = [' '.join(w) for w in test_words]

# 3. 创建特征
print("\n【3】创建特征")

# Word2Vec
print("Word2Vec...")
w2v = Word2Vec(sentences=all_words, vector_size=200, window=8, min_count=30, epochs=8, workers=4)

def get_w2v_features(word_lists):
    features = []
    for words in word_lists:
        vec = np.zeros(200, dtype='float32')
        count = 0
        for word in words:
            if word in w2v.wv:
                vec += w2v.wv[word]
                count += 1
        if count > 0:
            vec /= count
        features.append(vec)
    return np.array(features, dtype='float32')

X_train_w2v = get_w2v_features(train_words)
X_test_w2v = get_w2v_features(test_words)

# TF-IDF
print("TF-IDF...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=3, max_df=0.9, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_texts).toarray()
X_test_tfidf = tfidf.transform(test_texts).toarray()

# 合并特征
X_train = np.hstack([X_train_w2v, X_train_tfidf])
X_test = np.hstack([X_test_w2v, X_test_tfidf])

print(f"特征维度：{X_train.shape[1]}")

# 4. 训练模型
print("\n【4】训练模型")
clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# 5. 预测
print("\n【5】生成预测")
y_pred_train = clf.predict_proba(X_train)[:, 1]
y_pred = clf.predict_proba(X_test)[:, 1]

oof_auc = roc_auc_score(y_train, y_pred_train)
print(f"训练集 AUC: {oof_auc:.6f}")

# 6. 保存
print("\n【6】保存文件")
submission = pd.DataFrame({'id': test_df['id'], 'sentiment': y_pred})
submission_path = os.path.join(SUBMISSION_DIR, 'submission_improved.csv')
submission.to_csv(submission_path, index=False, quoting=0)

print(f"提交文件：{submission_path}")
print(f"样本数：{len(submission)}")
print(f"概率：[{y_pred.min():.4f}, {y_pred.max():.4f}]")
print(f"训练集 AUC: {oof_auc:.6f}")

# 保存统计
stats = {'训练集 AUC': oof_auc, '训练集': len(train_words), '测试集': len(test_words), '特征数': X_train.shape[1]}
pd.DataFrame(list(stats.items()), columns=['指标', '数值']).to_csv(
    os.path.join(RESULTS_DIR, 'improved_stats.csv'), index=False, encoding='utf-8-sig'
)

print("\n" + "=" * 70)
print(f"完成！训练集 AUC: {oof_auc:.6f}")
print("改进版提交文件已生成:", submission_path)
print("=" * 70)
