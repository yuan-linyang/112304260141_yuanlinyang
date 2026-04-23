# -*- coding: utf-8 -*-
"""
高级集成模型 - 目标 AUC 0.94+
作者：袁琳洋
学号：112304260141

改进策略:
1. 多随机种子 OOF 融合
2. Word2Vec + FastText + Doc2Vec 多语义特征
3. Stacking 集成
"""

import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'd:\kaggle 词袋遇到\word2vec-nlp-tutorial'
SUBMISSION_DIR = os.path.join(DATA_DIR, 'submission')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEEDS = [42, 123, 456, 789, 2024]
N_SPLITS = 5

def clean_review(review):
    review_text = BeautifulSoup(review, 'html.parser').get_text()
    letters_only = re.sub(r"[^a-zA-Z]", " ", review_text)
    return letters_only.lower().split()

def preprocess_df(df):
    print(f"预处理 {len(df)} 条评论...")
    return [clean_review(r) for r in df['review']]

def create_w2v_features(word_lists, all_words, vector_size=300):
    print("训练 Word2Vec...")
    model = Word2Vec(sentences=all_words, vector_size=vector_size, window=10, min_count=20, epochs=15, workers=4)
    print(f"词汇表：{len(model.wv)}")
    
    features = []
    for i, words in enumerate(word_lists):
        if i % 5000 == 0:
            print(f"  进度：{i}/{len(word_lists)}")
        vec = np.zeros(vector_size, dtype='float32')
        count = 0
        for word in words:
            if word in model.wv:
                vec += model.wv[word]
                count += 1
        if count > 0:
            vec /= count
        features.append(vec)
    return np.array(features, dtype='float32')

def create_ft_features(word_lists, all_words, vector_size=300):
    print("训练 FastText...")
    model = FastText(sentences=all_words, vector_size=vector_size, window=10, min_count=20, epochs=15, workers=4)
    
    features = []
    for i, words in enumerate(word_lists):
        if i % 5000 == 0:
            print(f"  进度：{i}/{len(word_lists)}")
        vec = np.zeros(vector_size, dtype='float32')
        count = 0
        for word in words:
            if word in model.wv:
                vec += model.wv[word]
                count += 1
        if count > 0:
            vec /= count
        features.append(vec)
    return np.array(features, dtype='float32')

def create_d2v_features(word_lists, all_words, vector_size=300):
    print("训练 Doc2Vec...")
    tagged_docs = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(all_words)]
    model = Doc2Vec(documents=tagged_docs, vector_size=vector_size, min_count=20, epochs=15, workers=4)
    
    features = []
    for i, words in enumerate(word_lists):
        if i % 5000 == 0:
            print(f"  进度：{i}/{len(word_lists)}")
        features.append(model.infer_vector(words))
    return np.array(features, dtype='float32')

def create_tfidf_features(text_list, max_features=50000):
    print("创建 TF-IDF 特征...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=max_features, min_df=2, max_df=0.95, sublinear_tf=True)
    features = vectorizer.fit_transform(text_list).toarray()
    print(f"TF-IDF 维度：{features.shape}")
    return features

def create_stats_features(word_lists):
    print("创建统计特征...")
    features = []
    for words in word_lists:
        feat = [
            len(words),
            np.mean([len(w) for w in words]) if words else 0,
            len(set(words)) / len(words) if words else 0,
            sum(1 for w in words if len(w) > 6) / len(words) if words else 0
        ]
        features.append(feat)
    return np.array(features, dtype='float32')

def get_oof_predictions(X, y, X_test, model_class, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_train = np.zeros(len(y))
    oof_test = np.zeros(len(X_test))
    
    for train_idx, val_idx in skf.split(X, y):
        model = model_class()
        model.fit(X[train_idx], y[train_idx])
        oof_train[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        oof_test += model.predict_proba(X_test)[:, 1] / n_splits
    
    return oof_train, oof_test

def main():
    print("=" * 70)
    print("高级集成模型 - 目标 AUC 0.94+")
    print("作者：袁琳洋  学号：112304260141")
    print("=" * 70)
    
    # 加载数据
    print("\n【1】加载数据")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'labeledTrainData.tsv'), sep='\t', quoting=3)
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'testData.tsv'), sep='\t', quoting=3)
    unlabeled_df = pd.read_csv(os.path.join(DATA_DIR, 'unlabeledTrainData.tsv'), sep='\t', quoting=3)
    
    # 预处理
    print("\n【2】预处理")
    train_words = preprocess_df(train_df)
    test_words = preprocess_df(test_df)
    unlabeled_words = preprocess_df(unlabeled_df)
    all_words = train_words + unlabeled_words + test_words
    
    # 创建特征
    print("\n【3】创建特征")
    X_train_w2v = create_w2v_features(train_words, all_words)
    X_test_w2v = create_w2v_features(test_words, all_words)
    
    X_train_ft = create_ft_features(train_words, all_words)
    X_test_ft = create_ft_features(test_words, all_words)
    
    X_train_d2v = create_d2v_features(train_words, all_words)
    X_test_d2v = create_d2v_features(test_words, all_words)
    
    train_texts = [' '.join(w) for w in train_words]
    test_texts = [' '.join(w) for w in test_words]
    X_train_tfidf = create_tfidf_features(train_texts + test_texts)
    X_test_tfidf = X_train_tfidf[len(train_df):]
    X_train_tfidf = X_train_tfidf[:len(train_df)]
    
    X_train_stats = create_stats_features(train_words)
    X_test_stats = create_stats_features(test_words)
    
    X_train = np.hstack([X_train_w2v, X_train_ft, X_train_d2v, X_train_tfidf, X_train_stats])
    X_test = np.hstack([X_test_w2v, X_test_ft, X_test_d2v, X_test_tfidf, X_test_stats])
    y_train = train_df['sentiment'].values
    
    print(f"\n最终特征维度：{X_train.shape[1]}")
    
    # 多随机种子 OOF
    print("\n【4】多随机种子 OOF 融合")
    all_oof_train = []
    all_oof_test = []
    
    models = [
        lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        lambda: LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        lambda: RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        lambda: GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        lambda: ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=42)
    ]
    
    for seed in RANDOM_SEEDS:
        print(f"\n种子 {seed}:")
        for i, model_fn in enumerate(models):
            oof_train, oof_test = get_oof_predictions(X_train, y_train, X_test, model_fn, N_SPLITS, seed)
            all_oof_train.append(oof_train)
            all_oof_test.append(oof_test)
            print(f"  模型{i+1}完成")
    
    # Stacking
    print("\n【5】Stacking 集成")
    stacked_train = np.column_stack(all_oof_train)
    stacked_test = np.column_stack(all_oof_test)
    print(f"Stacking 维度：{stacked_train.shape}")
    
    meta_model = LogisticRegression(C=1.0, max_iter=1000)
    meta_model.fit(stacked_train, y_train)
    
    # 预测
    print("\n【6】生成预测")
    y_pred = meta_model.predict_proba(stacked_test)[:, 1]
    
    # 评估
    y_pred_train = meta_model.predict_proba(stacked_train)[:, 1]
    oof_auc = roc_auc_score(y_train, y_pred_train)
    print(f"\nOOF AUC: {oof_auc:.6f}")
    
    # 保存
    print("\n【7】保存文件")
    submission = pd.DataFrame({'id': test_df['id'], 'sentiment': y_pred})
    submission_path = os.path.join(SUBMISSION_DIR, 'submission_ensemble_0.94.csv')
    submission.to_csv(submission_path, index=False, quoting=0)
    
    print(f"提交文件：{submission_path}")
    print(f"样本数：{len(submission)}")
    print(f"概率范围：[{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"OOF AUC: {oof_auc:.6f}")
    
    # 保存统计
    stats = {
        'OOF AUC': oof_auc,
        '训练集': len(train_df),
        '测试集': len(test_df),
        '特征数': X_train.shape[1],
        '种子数': len(RANDOM_SEEDS),
        '模型数': len(models)
    }
    pd.DataFrame(list(stats.items()), columns=['指标', '数值']).to_csv(
        os.path.join(RESULTS_DIR, 'ensemble_stats.csv'), index=False, encoding='utf-8-sig'
    )
    
    print("\n" + "=" * 70)
    print(f"完成！OOF AUC: {oof_auc:.6f}")
    print("=" * 70)
    
    return submission, oof_auc

if __name__ == '__main__':
    submission, auc = main()
