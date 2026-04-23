# -*- coding: utf-8 -*-
"""
生成改进版提交文件
"""

import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec, FastText

DATA_DIR = r'd:\kaggle 词袋遇到\word2vec-nlp-tutorial'
SUBMISSION_DIR = os.path.join(DATA_DIR, 'submission')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def clean_review(review):
    text = BeautifulSoup(review, 'html.parser').get_text()
    return re.sub(r"[^a-zA-Z]", " ", text).lower().split()

print("=" * 70)
print("生成改进版提交文件")
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
unlabeled_words = [clean_review(r) for r in unlabeled_df['review']]
all_words = train_words + unlabeled_words + test_words

y_train = train_df['sentiment'].values
train_texts = [' '.join(w) for w in train_words]
test_texts = [' '.join(w) for w in test_words]

# 3. 创建特征
print("\n【3】创建特征")

# Word2Vec
print("Word2Vec...")
w2v = Word2Vec(sentences=all_words, vector_size=300, window=10, min_count=20, epochs=10, workers=4)

def get_w2v_features(word_lists):
    features = []
    for words in word_lists:
        vec = np.zeros(300, dtype='float32')
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

# FastText
print("FastText...")
ft = FastText(sentences=all_words, vector_size=300, window=10, min_count=20, epochs=10, workers=4)

def get_ft_features(word_lists):
    features = []
    for words in word_lists:
        vec = np.zeros(300, dtype='float32')
        count = 0
        for word in words:
            if word in ft.wv:
                vec += ft.wv[word]
                count += 1
        if count > 0:
            vec /= count
        features.append(vec)
    return np.array(features, dtype='float32')

X_train_ft = get_ft_features(train_words)
X_test_ft = get_ft_features(test_words)

# TF-IDF
print("TF-IDF...")
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=30000, min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_texts + test_texts).toarray()
X_test_tfidf = X_train_tfidf[len(train_words):]
X_train_tfidf = X_train_tfidf[:len(train_words)]

# 合并特征
X_train = np.hstack([X_train_w2v, X_train_ft, X_train_tfidf])
X_test = np.hstack([X_test_w2v, X_test_ft, X_test_tfidf])

print(f"特征维度：{X_train.shape[1]}")

# 4. 多模型 OOF 集成
print("\n【4】多模型 OOF 集成")
seeds = [42, 123, 456]
models = [
    lambda s: LogisticRegression(C=1.0, max_iter=1000, random_state=s),
    lambda s: LogisticRegression(C=2.0, max_iter=1000, random_state=s),
    lambda s: RandomForestClassifier(n_estimators=100, max_depth=12, random_state=s),
    lambda s: GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=s)
]

all_oof_train = []
all_oof_test = []

for seed in seeds:
    print(f"\n种子 {seed}:")
    for i, model_fn in enumerate(models):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_train = np.zeros(len(y_train))
        oof_test = np.zeros(len(X_test))
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            model = model_fn(seed)
            model.fit(X_train[train_idx], y_train[train_idx])
            oof_train[val_idx] = model.predict_proba(X_train[val_idx])[:, 1]
            oof_test += model.predict_proba(X_test)[:, 1] / 5
        
        all_oof_train.append(oof_train)
        all_oof_test.append(oof_test)
        print(f"  模型{i+1}完成")

# 5. Stacking
print("\n【5】Stacking 集成")
stacked_train = np.column_stack(all_oof_train)
stacked_test = np.column_stack(all_oof_test)

meta_model = LogisticRegression(C=1.0, max_iter=1000)
meta_model.fit(stacked_train, y_train)

y_pred_train = meta_model.predict_proba(stacked_train)[:, 1]
y_pred = meta_model.predict_proba(stacked_test)[:, 1]

oof_auc = roc_auc_score(y_train, y_pred_train)
print(f"\nOOF AUC: {oof_auc:.6f}")

# 6. 保存
print("\n【6】保存文件")
submission = pd.DataFrame({'id': test_df['id'], 'sentiment': y_pred})
submission_path = os.path.join(SUBMISSION_DIR, 'submission_ensemble_improved.csv')
submission.to_csv(submission_path, index=False, quoting=0)

print(f"提交文件：{submission_path}")
print(f"样本数：{len(submission)}")
print(f"概率：[{y_pred.min():.4f}, {y_pred.max():.4f}]")
print(f"OOF AUC: {oof_auc:.6f}")

# 保存统计
stats = {'OOF AUC': oof_auc, '训练集': len(train_words), '测试集': len(test_words), '特征数': X_train.shape[1]}
pd.DataFrame(list(stats.items()), columns=['指标', '数值']).to_csv(
    os.path.join(RESULTS_DIR, 'ensemble_improved_stats.csv'), index=False, encoding='utf-8-sig'
)

print("\n" + "=" * 70)
print(f"完成！OOF AUC: {oof_auc:.6f}")
print("改进版提交文件已生成:", submission_path)
print("=" * 70)
