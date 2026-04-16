# -*- coding: utf-8 -*-
"""
基于 Word2Vec 的电影评论情感分析
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
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
DATA_DIR = r'd:\kaggle 词袋遇到\word2vec-nlp-tutorial'
CODE_DIR = os.path.join(DATA_DIR, 'code')
SUBMISSION_DIR = os.path.join(DATA_DIR, 'submission')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

# 确保目录存在
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Word2Vec 参数
WORD2VEC_PARAMS = {
    'vector_size': 300,      # 词向量维度
    'window': 10,            # 上下文窗口
    'min_count': 40,         # 最小词频
    'epochs': 10,            # 训练轮数
    'workers': 4,            # 并行线程数
    'sg': 1,                 # Skip-gram 架构
    'hs': 1,                 # Hierarchical Softmax
    'cbow': 0                # 不使用 CBOW
}

# TF-IDF 参数
TFIDF_PARAMS = {
    'ngram_range': (1, 4),
    'max_features': 100000,
    'min_df': 2,
    'max_df': 0.95,
    'sublinear_tf': True
}

# KMeans 聚类数
KMeans_CLUSTERS = 10

# ==================== 数据加载与预处理 ====================

def load_data(file_path):
    """加载数据文件"""
    print(f"加载数据：{file_path}")
    df = pd.read_csv(file_path, sep='\t', quoting=3)
    print(f"  - 样本数：{len(df)}")
    return df

def extract_zip(zip_path):
    """解压 zip 文件"""
    if not os.path.exists(zip_path):
        print(f"文件不存在：{zip_path}")
        return None
    
    extract_dir = os.path.dirname(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # 返回解压后的文件路径
    extracted_file = zip_path.replace('.zip', '')
    print(f"解压完成：{extracted_file}")
    return extracted_file

def review_to_words(review):
    """
    将影评文本转换为单词列表
    步骤：
    1. 去除 HTML 标签
    2. 只保留字母
    3. 转为小写并分割
    4. 去除停用词 (可选)
    """
    # 去除 HTML 标签
    review_text = BeautifulSoup(review, 'html.parser').get_text()
    
    # 只保留字母 (去除数字和标点)
    letters_only = re.sub(r"[^a-zA-Z]", " ", review_text)
    
    # 转为小写并分割成单词
    words = letters_only.lower().split()
    
    return words

def clean_review(review):
    """
    清洗单条评论
    返回清洗后的字符串 (用于 TF-IDF)
    """
    words = review_to_words(review)
    return ' '.join(words)

def preprocess_data(df, text_column='review'):
    """
    预处理数据框中的所有评论
    返回清洗后的文本列表和词列表
    """
    print(f"预处理 {len(df)} 条评论...")
    
    cleaned_reviews = []
    word_lists = []
    
    for i, review in enumerate(df[text_column]):
        if i % 1000 == 0:
            print(f"  处理进度：{i}/{len(df)}")
        
        # 获取词列表
        words = review_to_words(review)
        word_lists.append(words)
        
        # 获取清洗后的字符串
        cleaned = ' '.join(words)
        cleaned_reviews.append(cleaned)
    
    print("预处理完成!")
    return cleaned_reviews, word_lists

# ==================== Word2Vec 模型训练 ====================

def train_word2vec(word_lists, model_path=None):
    """
    训练 Word2Vec 模型
    """
    print("开始训练 Word2Vec 模型...")
    print(f"  语料库大小：{len(word_lists)} 条句子")
    
    # 训练模型
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
    
    print(f"Word2Vec 训练完成!")
    print(f"  词汇表大小：{len(model.wv)}")
    
    # 保存模型
    if model_path:
        model.save(model_path)
        print(f"  模型已保存：{model_path}")
    
    return model

def load_word2vec_model(model_path):
    """加载 Word2Vec 模型"""
    if os.path.exists(model_path):
        print(f"加载已有模型：{model_path}")
        return Word2Vec.load(model_path)
    return None

# ==================== 特征提取 ====================

def review_to_vector(review_words, model, vector_size):
    """
    将评论转换为平均词向量
    """
    feature_vector = np.zeros(vector_size, dtype='float32')
    word_count = 0
    
    for word in review_words:
        if word in model.wv:
            feature_vector += model.wv[word]
            word_count += 1
    
    if word_count > 0:
        feature_vector /= word_count
    
    return feature_vector

def create_mean_embeddings(word_lists, model):
    """
    创建平均词向量特征
    """
    print("创建平均词向量特征...")
    
    n_sentences = len(word_lists)
    vector_size = WORD2VEC_PARAMS['vector_size']
    
    feature_matrix = np.zeros((n_sentences, vector_size), dtype='float32')
    
    for i, words in enumerate(word_lists):
        if i % 1000 == 0:
            print(f"  处理进度：{i}/{n_sentences}")
        feature_matrix[i] = review_to_vector(words, model, vector_size)
    
    print("平均词向量特征创建完成!")
    return feature_matrix

def create_tfidf_features(cleaned_reviews, train_size):
    """
    创建 TF-IDF 特征
    """
    print("创建 TF-IDF 特征...")
    
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_PARAMS['ngram_range'],
        max_features=TFIDF_PARAMS['max_features'],
        min_df=TFIDF_PARAMS['min_df'],
        max_df=TFIDF_PARAMS['max_df'],
        sublinear_tf=TFIDF_PARAMS['sublinear_tf']
    )
    
    # 只在训练集上拟合
    train_vectorizer = vectorizer.fit(cleaned_reviews[:train_size])
    
    # 转换所有数据
    tfidf_matrix = train_vectorizer.transform(cleaned_reviews)
    
    print(f"TF-IDF 特征维度：{tfidf_matrix.shape}")
    return tfidf_matrix, train_vectorizer

# ==================== 模型训练与评估 ====================

def evaluate_model(model, X, y, n_splits=5):
    """
    使用分层 K 折交叉验证评估模型
    """
    print(f"  使用 {n_splits} 折交叉验证评估...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    
    print(f"  ROC-AUC 得分：{scores}")
    print(f"  平均 ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return scores.mean(), scores.std()

def compare_classifiers(X_train, y_train):
    """
    比较不同分类器的效果
    """
    print("\n比较不同分类器的效果:")
    print("=" * 60)
    
    classifiers = {
        'Logistic Regression': LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'LinearSVC': LinearSVC(C=1.0, max_iter=1000, random_state=42)
    }
    
    results = []
    
    for name, clf in classifiers.items():
        print(f"\n模型：{name}")
        mean_score, std_score = evaluate_model(clf, X_train, y_train)
        results.append({
            'model': name,
            'mean_auc': mean_score,
            'std_auc': std_score
        })
    
    # 创建结果 DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_auc', ascending=False)
    
    print("\n" + "=" * 60)
    print("模型比较结果:")
    print(results_df.to_string(index=False))
    
    # 保存结果
    results_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n结果已保存：{results_path}")
    
    return results_df

# ==================== 主流程 ====================

def main():
    """
    主流程
    """
    print("=" * 60)
    print("基于 Word2Vec 的电影评论情感分析")
    print("作者：袁琳洋  学号：112304260141")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n【步骤 1】加载数据")
    labeled_path = extract_zip(os.path.join(DATA_DIR, 'labeledTrainData.tsv.zip'))
    test_path = extract_zip(os.path.join(DATA_DIR, 'testData.tsv.zip'))
    unlabeled_path = extract_zip(os.path.join(DATA_DIR, 'unlabeledTrainData.tsv.zip'))
    
    train_df = load_data(labeled_path)
    test_df = load_data(test_path)
    unlabeled_df = load_data(unlabeled_path)
    
    # 2. 数据预处理
    print("\n【步骤 2】数据预处理")
    print("处理训练集...")
    train_cleaned, train_words = preprocess_data(train_df)
    
    print("\n处理测试集...")
    test_cleaned, test_words = preprocess_data(test_df)
    
    print("\n处理无标签数据...")
    unlabeled_cleaned, unlabeled_words = preprocess_data(unlabeled_df)
    
    # 3. 训练 Word2Vec 模型
    print("\n【步骤 3】训练 Word2Vec 模型")
    model_path = os.path.join(CODE_DIR, 'word2vec_model.model')
    
    # 合并所有语料
    all_words = train_words + unlabeled_words + test_words
    print(f"总语料：{len(all_words)} 条句子")
    
    # 训练或加载模型
    model = load_word2vec_model(model_path)
    if model is None:
        model = train_word2vec(all_words, model_path)
    
    # 4. 创建特征
    print("\n【步骤 4】创建特征")
    
    # 平均词向量特征
    X_train_mean = create_mean_embeddings(train_words, model)
    X_test_mean = create_mean_embeddings(test_words, model)
    
    y_train = train_df['sentiment'].values
    
    # 5. 模型比较
    print("\n【步骤 5】模型比较")
    results_df = compare_classifiers(X_train_mean, y_train)
    
    # 6. 选择最优模型
    print("\n【步骤 6】训练最优模型")
    best_model_name = results_df.iloc[0]['model']
    print(f"最优模型：{best_model_name}")
    
    # 根据实验流程，使用 Logistic Regression(C=2.0)
    final_model = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    print("在全训练集上训练最终模型...")
    final_model.fit(X_train_mean, y_train)
    print("训练完成!")
    
    # 7. 预测测试集
    print("\n【步骤 7】预测测试集")
    y_pred_proba = final_model.predict_proba(X_test_mean)[:, 1]
    
    # 8. 生成提交文件
    print("\n【步骤 8】生成提交文件")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': y_pred_proba
    })
    
    submission_path = os.path.join(SUBMISSION_DIR, 'submission_word2vec_mean_lr.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"提交文件已保存：{submission_path}")
    print(f"  - 提交样本数：{len(submission_df)}")
    print(f"  - 预测概率范围：[{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    
    # 9. 保存实验结果
    print("\n【步骤 9】保存实验结果")
    
    # 特征统计
    feature_stats = {
        '训练集样本数': len(train_df),
        '测试集样本数': len(test_df),
        '无标签样本数': len(unlabeled_df),
        '词向量维度': WORD2VEC_PARAMS['vector_size'],
        '词汇表大小': len(model.wv),
        '最优模型': best_model_name,
        '最佳 ROC-AUC': results_df.iloc[0]['mean_auc']
    }
    
    stats_df = pd.DataFrame(list(feature_stats.items()), columns=['指标', '数值'])
    stats_path = os.path.join(RESULTS_DIR, 'experiment_stats.csv')
    stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"实验统计已保存：{stats_path}")
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)
    
    return submission_path, results_df

if __name__ == '__main__':
    submission_file, results = main()
