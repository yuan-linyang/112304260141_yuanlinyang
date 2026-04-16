# -*- coding: utf-8 -*-
"""
基于 Word2Vec 的电影评论情感分析
作者：袁琳洋
学号：112304260141
"""

import os
import re
import zipfile
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'd:\kaggle 词袋遇到\word2vec-nlp-tutorial'

def clean_review(review):
    """清洗评论"""
    review_text = BeautifulSoup(review, 'html.parser').get_text()
    letters_only = re.sub(r"[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    return words

def main():
    print("加载数据...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'labeledTrainData.tsv'), sep='\t', quoting=3)
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'testData.tsv'), sep='\t', quoting=3)
    unlabeled_df = pd.read_csv(os.path.join(DATA_DIR, 'unlabeledTrainData.tsv'), sep='\t', quoting=3)
    
    print(f"训练集：{len(train_df)}, 测试集：{len(test_df)}, 无标签：{len(unlabeled_df)}")
    
    # 预处理
    print("\n预处理...")
    train_words = [clean_review(r) for r in train_df['review']]
    test_words = [clean_review(r) for r in test_df['review']]
    unlabeled_words = [clean_review(r) for r in unlabeled_df['review']]
    
    # 训练 Word2Vec
    print("\n训练 Word2Vec...")
    all_words = train_words + unlabeled_words + test_words
    model = Word2Vec(sentences=all_words, vector_size=100, window=5, min_count=100, workers=4, sg=1, hs=1, epochs=5)
    print(f"词汇表：{len(model.wv)}")
    
    # 创建特征
    print("\n创建特征...")
    def create_features(word_lists):
        features = []
        for i, words in enumerate(word_lists):
            vec = np.zeros(100)
            count = 0
            for word in words:
                if word in model.wv:
                    vec += model.wv[word]
                    count += 1
            if count > 0:
                vec /= count
            features.append(vec)
        return np.array(features)
    
    X_train = create_features(train_words)
    X_test = create_features(test_words)
    y_train = train_df['sentiment'].values
    
    # 训练模型
    print("\n训练 Logistic Regression...")
    clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # 预测
    print("\n预测...")
    y_pred = clf.predict_proba(X_test)[:, 1]
    
    # 保存
    submission = pd.DataFrame({'id': test_df['id'], 'sentiment': y_pred})
    submission.to_csv(os.path.join(DATA_DIR, 'submission', 'submission_word2vec_mean_lr.csv'), index=False)
    print(f"\n提交文件已保存：{len(submission)} 条")
    print(f"概率范围：[{y_pred.min():.4f}, {y_pred.max():.4f}]")

if __name__ == '__main__':
    main()
