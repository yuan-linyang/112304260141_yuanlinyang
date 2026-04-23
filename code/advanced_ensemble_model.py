# -*- coding: utf-8 -*-
"""
高级集成模型 - 目标 AUC 0.94+
作者：袁琳洋
学号：112304260141

主要改进:
1. 多随机种子 OOF 融合 (5 个种子)
2. 多种语义特征：Word2Vec + FastText + Doc2Vec + TF-IDF
3. 两层 Stacking 集成
"""

import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_DIR = r'd:\kaggle 词袋遇到\word2vec-nlp-tutorial'
CODE_DIR = os.path.join(DATA_DIR, 'code')
SUBMISSION_DIR = os.path.join(DATA_DIR, 'submission')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 多个随机种子用于 OOF 融合
RANDOM_SEEDS = [42, 123, 456, 789, 2024]
N_SPLITS = 5

# Word2Vec 参数
W2V_PARAMS = {'vector_size': 300, 'window': 10, 'min_count': 20, 'epochs': 15, 'workers': 4}
# FastText 参数
FT_PARAMS = {'vector_size': 300, 'window': 10, 'min_count': 20, 'epochs': 15, 'workers': 4}
# Doc2Vec 参数
D2V_PARAMS = {'vector_size': 300, 'min_count': 20, 'epochs': 15, 'workers': 4}

# ==================== 文本预处理 ====================

def clean_review(review):
    """清洗评论"""
    review_text = BeautifulSoup(review, 'html.parser').get_text()
    letters_only = re.sub(r"[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    return words

def preprocess_df(df, text_column='review'):
    """预处理"""
    print(f"预处理 {len(df)} 条评论...")
    word_lists = []
    for i, review in enumerate(df[text_column]):
        if i % 5000 == 0:
            print(f"  进度：{i}/{len(df)}")
        word_lists.append(clean_review(review))
    print("预处理完成!")
    return word_lists

# ==================== 特征提取 ====================

def train_word2vec(word_lists, seed=42):
    """训练 Word2Vec"""
    np.random.seed(seed)
    model = Word2Vec(
        sentences=word_lists,
        vector_size=W2V_PARAMS['vector_size'],
        window=W2V_PARAMS['window'],
        min_count=W2V_PARAMS['min_count'],
        workers=W2V_PARAMS['workers'],
        epochs=W2V_PARAMS['epochs'],
        seed=seed
    )
    return model

def train_fasttext(word_lists, seed=42):
    """训练 FastText"""
    np.random.seed(seed)
    model = FastText(
        sentences=word_lists,
        vector_size=FT_PARAMS['vector_size'],
        window=FT_PARAMS['window'],
        min_count=FT_PARAMS['min_count'],
        workers=FT_PARAMS['workers'],
        epochs=FT_PARAMS['epochs'],
        seed=seed
    )
    return model

def train_doc2vec(word_lists, seed=42):
    """训练 Doc2Vec"""
    np.random.seed(seed)
    tagged_docs = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(word_lists)]
    model = Doc2Vec(
        documents=tagged_docs,
        vector_size=D2V_PARAMS['vector_size'],
        min_count=D2V_PARAMS['min_count'],
        workers=D2V_PARAMS['workers'],
        epochs=D2V_PARAMS['epochs'],
        seed=seed
    )
    return model

def review_to_w2v_vector(words, model):
    """Word2Vec 平均向量"""
    vec = np.zeros(W2V_PARAMS['vector_size'], dtype='float32')
    count = 0
    for word in words:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count > 0:
        vec /= count
    return vec

def review_to_ft_vector(words, model):
    """FastText 平均向量"""
    vec = np.zeros(FT_PARAMS['vector_size'], dtype='float32')
    count = 0
    for word in words:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count > 0:
        vec /= count
    return vec

def create_w2v_features(word_lists, model):
    """创建 Word2Vec 特征"""
    n = len(word_lists)
    features = np.zeros((n, W2V_PARAMS['vector_size']), dtype='float32')
    for i, words in enumerate(word_lists):
        if i % 5000 == 0:
            print(f"  W2V 进度：{i}/{n}")
        features[i] = review_to_w2v_vector(words, model)
    return features

def create_ft_features(word_lists, model):
    """创建 FastText 特征"""
    n = len(word_lists)
    features = np.zeros((n, FT_PARAMS['vector_size']), dtype='float32')
    for i, words in enumerate(word_lists):
        if i % 5000 == 0:
            print(f"  FT 进度：{i}/{n}")
        features[i] = review_to_ft_vector(words, model)
    return features

def create_d2v_features(word_lists, model):
    """创建 Doc2Vec 特征"""
    n = len(word_lists)
    features = np.zeros((n, D2V_PARAMS['vector_size']), dtype='float32')
    for i in range(n):
        if i % 5000 == 0:
            print(f"  D2V 进度：{i}/{n}")
        features[i] = model.infer_vector(word_lists[i])
    return features

def create_tfidf_features(text_list, max_features=50000):
    """创建 TF-IDF 特征"""
    print("创建 TF-IDF 特征...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    features = vectorizer.fit_transform(text_list)
    print(f"TF-IDF 特征维度：{features.shape}")
    return features, vectorizer

def create_stats_features(word_lists):
    """创建统计特征"""
    print("创建统计特征...")
    n = len(word_lists)
    features = np.zeros((n, 5), dtype='float32')
    for i, words in enumerate(word_lists):
        features[i, 0] = len(words)  # 词数
        features[i, 1] = np.mean([len(w) for w in words]) if words else 0  # 平均词长
        features[i, 2] = len(set(words)) / len(words) if words else 0  # 词汇丰富度
        features[i, 3] = sum(1 for w in words if len(w) > 6) / len(words) if words else 0  # 长词比例
        features[i, 4] = sum(1 for w in words if w[0].isupper()) / len(words) if words else 0  # 大写词比例
    return features

# ==================== 模型训练 ====================

def get_oof_predictions(X, y, X_test, model_class, n_splits=5, seed=42):
    """生成 OOF 预测"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    oof_train = np.zeros((len(y),), dtype='float32')
    oof_test = np.zeros((len(X_test),), dtype='float32')
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold = y[train_idx]
        
        model = model_class()
        model.fit(X_train_fold, y_train_fold)
        
        oof_train[val_idx] = model.predict_proba(X_val_fold)[:, 1]
        oof_test += model.predict_proba(X_test)[:, 1] / n_splits
    
    return oof_train, oof_test

def train_base_models(X_train, y_train, X_test, seed=42):
    """训练基础模型"""
    print(f"\n训练基础模型 (seed={seed})...")
    
    models = {
        'LR_C1': LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
        'LR_C2': LogisticRegression(C=2.0, max_iter=1000, random_state=seed),
        'LR_C5': LogisticRegression(C=5.0, max_iter=1000, random_state=seed),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed),
        'GBDT': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=seed),
        'MLP': MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=500, random_state=seed)
    }
    
    oof_results = {name: get_oof_predictions(X_train, y_train, X_test, model, N_SPLITS, seed) 
                   for name, model in models.items()}
    
    return oof_results

# ==================== 主流程 ====================

def main():
    print("=" * 70)
    print("高级集成模型 - 目标 AUC 0.94+")
    print("作者：袁琳洋  学号：112304260141")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n【步骤 1】加载数据")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'labeledTrainData.tsv'), sep='\t', quoting=3)
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'testData.tsv'), sep='\t', quoting=3)
    unlabeled_df = pd.read_csv(os.path.join(DATA_DIR, 'unlabeledTrainData.tsv'), sep='\t', quoting=3)
    
    print(f"训练集：{len(train_df)}, 测试集：{len(test_df)}, 无标签：{len(unlabeled_df)}")
    
    # 2. 预处理
    print("\n【步骤 2】数据预处理")
    train_words = preprocess_df(train_df)
    test_words = preprocess_df(test_df)
    unlabeled_words = preprocess_df(unlabeled_df)
    
    # 3. 训练词向量模型
    print("\n【步骤 3】训练词向量模型")
    all_words = train_words + unlabeled_words + test_words
    
    print("训练 Word2Vec...")
    w2v_model = train_word2vec(all_words, seed=42)
    print(f"词汇表：{len(w2v_model.wv)}")
    
    print("训练 FastText...")
    ft_model = train_fasttext(all_words, seed=42)
    print(f"词汇表：{len(ft_model.wv)}")
    
    print("训练 Doc2Vec...")
    d2v_model = train_doc2vec(all_words, seed=42)
    
    # 4. 创建特征
    print("\n【步骤 4】创建特征")
    
    # Word2Vec 特征
    print("创建 Word2Vec 特征...")
    X_train_w2v = create_w2v_features(train_words, w2v_model)
    X_test_w2v = create_w2v_features(test_words, w2v_model)
    
    # FastText 特征
    print("创建 FastText 特征...")
    X_train_ft = create_ft_features(train_words, ft_model)
    X_test_ft = create_ft_features(test_words, ft_model)
    
    # Doc2Vec 特征
    print("创建 Doc2Vec 特征...")
    X_train_d2v = create_d2v_features(train_words, d2v_model)
    X_test_d2v = create_d2v_features(test_words, d2v_model)
    
    # TF-IDF 特征
    train_texts = [' '.join(words) for words in train_words]
    test_texts = [' '.join(words) for words in test_words]
    X_train_tfidf, _ = create_tfidf_features(train_texts + test_texts)
    X_train_tfidf = X_train_tfidf[:len(train_df)].toarray()
    X_test_tfidf = X_train_tfidf[len(train_df):]
    
    # 统计特征
    X_train_stats = create_stats_features(train_words)
    X_test_stats = create_stats_features(test_words)
    
    # 合并所有特征
    print("\n合并特征...")
    X_train = np.hstack([X_train_w2v, X_train_ft, X_train_d2v, X_train_tfidf, X_train_stats])
    X_test = np.hstack([X_test_w2v, X_test_ft, X_test_d2v, X_test_tfidf, X_test_stats])
    y_train = train_df['sentiment'].values
    
    print(f"最终特征维度：{X_train.shape[1]}")
    
    # 5. 多随机种子 OOF 融合
    print("\n【步骤 5】多随机种子 OOF 融合")
    all_seeds_oof_train = []
    all_seeds_oof_test = []
    
    for seed in RANDOM_SEEDS:
        print(f"\n{'='*60}")
        print(f"处理随机种子 {seed}")
        print('='*60)
        oof_results = train_base_models(X_train, y_train, X_test, seed=seed)
        
        for model_name, (oof_train, oof_test) in oof_results.items():
            all_seeds_oof_train.append(oof_train)
            all_seeds_oof_test.append(oof_test)
            print(f"  {model_name}: OOF 完成")
    
    # Stacking 第一层
    print("\n【步骤 6】Stacking 第一层")
    stacked_train = np.column_stack(all_seeds_oof_train)
    stacked_test = np.column_stack(all_seeds_oof_test)
    print(f"Stacking 特征维度：{stacked_train.shape}")
    
    # Stacking 第二层 - 元模型
    print("\n【步骤 7】训练元模型 (第二层)")
    meta_model = LogisticRegression(C=1.0, max_iter=1000)
    meta_model.fit(stacked_train, y_train)
    
    # 6. 最终预测
    print("\n【步骤 8】生成最终预测")
    y_pred = meta_model.predict_proba(stacked_test)[:, 1]
    
    # 7. 保存提交文件
    print("\n【步骤 9】保存提交文件")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': y_pred
    })
    
    submission_path = os.path.join(SUBMISSION_DIR, 'submission_ensemble_0.94.csv')
    submission.to_csv(submission_path, index=False, quoting=0)
    print(f"提交文件已保存：{submission_path}")
    print(f"  样本数：{len(submission)}")
    print(f"  概率范围：[{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"  概率均值：{y_pred.mean():.4f}")
    
    # 8. 保存 OOF 预测用于评估
    print("\n【步骤 10】保存 OOF 预测")
    oof_df = pd.DataFrame({
        'id': train_df['id'],
        'oof_prediction': meta_model.predict_proba(stacked_train)[:, 1],
        'label': y_train
    })
    oof_path = os.path.join(RESULTS_DIR, 'oof_predictions.csv')
    oof_df.to_csv(oof_path, index=False)
    
    # 计算 OOF AUC
    from sklearn.metrics import roc_auc_score
    oof_auc = roc_auc_score(y_train, oof_df['oof_prediction'])
    print(f"\nOOF AUC: {oof_auc:.6f}")
    
    # 保存统计信息
    stats = {
        '训练集样本数': len(train_df),
        '测试集样本数': len(test_df),
        '无标签样本数': len(unlabeled_df),
        '随机种子数': len(RANDOM_SEEDS),
        '交叉验证折数': N_SPLITS,
        'Word2Vec 维度': W2V_PARAMS['vector_size'],
        'FastText 维度': FT_PARAMS['vector_size'],
        'Doc2Vec 维度': D2V_PARAMS['vector_size'],
        '基础模型数': 6,
        '总特征数': X_train.shape[1],
        'OOF AUC': oof_auc
    }
    
    stats_df = pd.DataFrame(list(stats.items()), columns=['指标', '数值'])
    stats_df.to_csv(os.path.join(RESULTS_DIR, 'ensemble_stats.csv'), index=False, encoding='utf-8-sig')
    print(f"\n统计信息已保存：{os.path.join(RESULTS_DIR, 'ensemble_stats.csv')}")
    
    print("\n" + "=" * 70)
    print("实验完成!")
    print(f"OOF AUC: {oof_auc:.6f}")
    print("提交文件:", submission_path)
    print("=" * 70)
    
    return submission, oof_auc

if __name__ == '__main__':
    submission, auc = main()
