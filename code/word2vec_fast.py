# -*- coding: utf-8 -*-
"""
基于 Word2Vec 的电影评论情感分析 - 快速版本
作者：袁琳洋
学号：112304260141
班级：数据 1231
"""

import os
import re
import zipfile
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
DATA_DIR = r'd:\kaggle 词袋遇到\word2vec-nlp-tutorial'
CODE_DIR = os.path.join(DATA_DIR, 'code')
SUBMISSION_DIR = os.path.join(DATA_DIR, 'submission')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Word2Vec 参数 (简化版)
WORD2VEC_PARAMS = {
    'vector_size': 100,
    'window': 5,
    'min_count': 100,
    'epochs': 5,
    'workers': 4,
    'sg': 1,
    'hs': 1
}

# ==================== 数据加载 ====================

def load_and_extract(file_path):
    """加载并解压数据"""
    print(f"处理：{file_path}")
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))
        file_path = file_path.replace('.zip', '')
    
    df = pd.read_csv(file_path, sep='\t', quoting=3)
    print(f"  样本数：{len(df)}")
    return df

# ==================== 文本预处理 ====================

def clean_review(review):
    """清洗单条评论"""
    # 去除 HTML
    review_text = BeautifulSoup(review, 'html.parser').get_text()
    # 只保留字母
    letters_only = re.sub(r"[^a-zA-Z]", " ", review_text)
    # 转小写并分割
    words = letters_only.lower().split()
    return ' '.join(words), words

def preprocess_df(df, text_column='review'):
    """预处理数据框"""
    print(f"预处理 {len(df)} 条评论...")
    cleaned_reviews = []
    word_lists = []
    
    for i, review in enumerate(df[text_column]):
        if i % 5000 == 0:
            print(f"  进度：{i}/{len(df)}")
        cleaned, words = clean_review(review)
        cleaned_reviews.append(cleaned)
        word_lists.append(words)
    
    print("预处理完成!")
    return cleaned_reviews, word_lists

# ==================== Word2Vec 训练 ====================

def train_word2vec_fast(word_lists):
    """快速训练 Word2Vec"""
    from gensim.models import Word2Vec
    
    print("训练 Word2Vec...")
    model = Word2Vec(
        sentences=word_lists,
        vector_size=WORD2VEC_PARAMS['vector_size'],
        window=WORD2VEC_PARAMS['window'],
        min_count=WORD2VEC_PARAMS['min_count'],
        workers=WORD2VEC_PARAMS['workers'],
        sg=WORD2VEC_PARAMS['sg'],
        hs=WORD2VEC_PARAMS['hs'],
        epochs=WORD2VEC_PARAMS['epochs']
    )
    
    print(f"词汇表大小：{len(model.wv)}")
    return model

# ==================== 特征提取 ====================

def review_to_vector(words, model):
    """将评论转换为平均词向量"""
    vector_size = WORD2VEC_PARAMS['vector_size']
    feature_vector = np.zeros(vector_size, dtype='float32')
    word_count = 0
    
    for word in words:
        if word in model.wv:
            feature_vector += model.wv[word]
            word_count += 1
    
    if word_count > 0:
        feature_vector /= word_count
    
    return feature_vector

def create_features(word_lists, model):
    """创建特征矩阵"""
    print("创建特征...")
    n_sentences = len(word_lists)
    vector_size = WORD2VEC_PARAMS['vector_size']
    
    feature_matrix = np.zeros((n_sentences, vector_size), dtype='float32')
    
    for i, words in enumerate(word_lists):
        if i % 5000 == 0:
            print(f"  进度：{i}/{n_sentences}")
        feature_matrix[i] = review_to_vector(words, model)
    
    print("特征创建完成!")
    return feature_matrix

# ==================== 模型评估 ====================

def evaluate_model(model, X, y, n_splits=3):
    """交叉验证评估"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    return scores.mean(), scores.std()

# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("基于 Word2Vec 的电影评论情感分析")
    print("作者：袁琳洋  学号：112304260141")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n【步骤 1】加载数据")
    train_df = load_and_extract(os.path.join(DATA_DIR, 'labeledTrainData.tsv.zip'))
    test_df = load_and_extract(os.path.join(DATA_DIR, 'testData.tsv.zip'))
    unlabeled_df = load_and_extract(os.path.join(DATA_DIR, 'unlabeledTrainData.tsv.zip'))
    
    # 2. 预处理
    print("\n【步骤 2】数据预处理")
    train_cleaned, train_words = preprocess_df(train_df)
    test_cleaned, test_words = preprocess_df(test_df)
    unlabeled_cleaned, unlabeled_words = preprocess_df(unlabeled_df)
    
    # 3. 训练 Word2Vec
    print("\n【步骤 3】训练 Word2Vec 模型")
    all_words = train_words + unlabeled_words + test_words
    model = train_word2vec_fast(all_words)
    
    # 4. 创建特征
    print("\n【步骤 4】创建特征")
    X_train = create_features(train_words, model)
    X_test = create_features(test_words, model)
    y_train = train_df['sentiment'].values
    
    # 5. 模型比较
    print("\n【步骤 5】模型比较")
    classifiers = {
        'Logistic Regression(C=2.0)': LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        'LinearSVC': LinearSVC(C=1.0, max_iter=500, random_state=42)
    }
    
    results = []
    for name, clf in classifiers.items():
        print(f"\n模型：{name}")
        mean_auc, std_auc = evaluate_model(clf, X_train, y_train)
        print(f"  ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
        results.append({'model': name, 'mean_auc': mean_auc, 'std_auc': std_auc})
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_auc', ascending=False)
    print("\n模型比较结果:")
    print(results_df.to_string(index=False))
    
    # 保存结果
    results_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
    
    # 6. 训练最优模型
    print("\n【步骤 6】训练最终模型")
    best_model = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    best_model.fit(X_train, y_train)
    print("训练完成!")
    
    # 7. 预测
    print("\n【步骤 7】预测测试集")
    y_pred = best_model.predict_proba(X_test)[:, 1]
    
    # 8. 生成提交文件
    print("\n【步骤 8】生成提交文件")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': y_pred
    })
    
    submission_path = os.path.join(SUBMISSION_DIR, 'submission_word2vec_mean_lr.csv')
    submission.to_csv(submission_path, index=False)
    print(f"提交文件已保存：{submission_path}")
    print(f"  样本数：{len(submission)}")
    print(f"  概率范围：[{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # 9. 保存统计信息
    print("\n【步骤 9】保存实验统计")
    stats = {
        '训练集样本数': len(train_df),
        '测试集样本数': len(test_df),
        '无标签样本数': len(unlabeled_df),
        '词向量维度': WORD2VEC_PARAMS['vector_size'],
        '词汇表大小': len(model.wv),
        '最优模型': 'Logistic Regression(C=2.0)',
        '最佳 ROC-AUC': results_df.iloc[0]['mean_auc']
    }
    
    stats_df = pd.DataFrame(list(stats.items()), columns=['指标', '数值'])
    stats_df.to_csv(os.path.join(RESULTS_DIR, 'experiment_stats.csv'), index=False, encoding='utf-8-sig')
    print(f"统计已保存：{os.path.join(RESULTS_DIR, 'experiment_stats.csv')}")
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)
    
    return submission, results_df

if __name__ == '__main__':
    submission, results = main()
