# 高级集成模型说明 - 目标 AUC 0.94+

## 作者信息
- **姓名**: 袁琳洋
- **学号**: 112304260141
- **班级**: 数据 1231

## 改进策略

### 1. 多随机种子 OOF 融合
使用 5 个不同的随机种子 (42, 123, 456, 789, 2024)，每个种子训练 5 折交叉验证，有效减少模型方差。

**优势**:
- 降低模型对特定数据划分的敏感性
- 提高模型泛化能力
- 减少过拟合风险

### 2. 多种语义特征融合

#### Word2Vec (300 维)
- 架构：Skip-gram
- 窗口大小：10
- 最小词频：20
- 训练轮数：15
- 特征：平均词向量

#### FastText (300 维)
- 考虑词内字符信息
- 更好地处理未登录词
- 参数与 Word2Vec 一致

#### Doc2Vec (300 维)
- 直接学习文档向量
- 捕捉文档级别语义
- 补充词向量信息

#### TF-IDF (30000-50000 维)
- N-gram 范围：(1, 3)
- 最大特征数：30000-50000
- Sublinear TF 缩放

#### 统计特征 (4-5 维)
- 评论长度
- 平均词长
- 词汇丰富度
- 长词比例

**总特征维度**: 约 1200-1500 维

### 3. 两层 Stacking 集成

#### 第一层 (Base Models)
每个随机种子训练 5-6 个基础模型:
1. Logistic Regression (C=1.0)
2. Logistic Regression (C=2.0)
3. Logistic Regression (C=5.0)
4. Random Forest (n_estimators=100, max_depth=15)
5. Gradient Boosting (n_estimators=100, max_depth=5)
6. Extra Trees (n_estimators=100, max_depth=15)

**总模型数**: 5 个种子 × 6 个模型 = 30 个 OOF 预测

#### 第二层 (Meta Model)
- 模型：Logistic Regression (C=1.0)
- 输入：30 个 OOF 预测
- 输出：最终预测概率

## 代码文件

### 完整版
- `code/advanced_ensemble_model.py` - 完整实现 (训练时间较长)
- `code/ensemble_v2.py` - 优化版本
- `code/ensemble_fast.py` - 快速版本 (使用部分数据加速)

### 基础版
- `code/word2vec_sentiment_analysis.py` - 基础 Word2Vec 模型
- `code/word2vec_fast.py` - 快速版本
- `code/generate_submission.py` - 提交文件生成

## 预期效果

### 基础模型 (Word2Vec + LR)
- OOF AUC: ~0.90-0.92
- Kaggle Public Score: ~0.88-0.90

### 高级集成模型
- OOF AUC: **0.94+**
- Kaggle Public Score: **0.92+**

## 使用方法

### 运行完整版
```python
python code/advanced_ensemble_model.py
```
训练时间：约 2-4 小时

### 运行快速版
```python
python code/ensemble_fast.py
```
训练时间：约 30-60 分钟

## 输出文件

### 提交文件
- `submission/submission_ensemble_0.94.csv` - 高级集成模型提交
- `submission/submission_ensemble_v2.csv` - 优化版提交
- `submission/submission_word2vec_mean_lr.csv` - 基础模型提交

### 统计信息
- `results/ensemble_stats.csv` - 模型统计
- `results/oof_predictions.csv` - OOF 预测 (用于本地评估)

## 关键参数调优

### 可调整参数
1. **随机种子数量**: 增加种子数可提高稳定性，但增加训练时间
2. **词向量维度**: 300 维通常足够，可尝试 200 或 400
3. **TF-IDF 特征数**: 30000-50000，越多越容易过拟合
4. **基础模型数量**: 可增加更多模型类型
5. **交叉验证折数**: 5 折是常用选择，可尝试 10 折

### 建议
- 先用快速版本验证流程
- 再用完整版获取最佳效果
- 根据 OOF AUC 调整参数
- 提交到 Kaggle 验证实际效果

## 注意事项

1. **训练时间**: 完整训练需要较长时间，建议使用完整数据集
2. **内存使用**: 多特征融合需要较多内存
3. **过拟合**: Stacking 层数不宜过多，避免过拟合
4. **随机性**: 多次运行结果可能略有差异

## 下一步改进

1. **更多特征**
   - GloVe 预训练词向量
   - BERT 等预训练语言模型特征
   - 情感词典特征

2. **更多模型**
   - XGBoost
   - LightGBM
   - CatBoost
   - Neural Networks

3. **更复杂集成**
   - Blending (留出验证集)
   - Multi-level Stacking
   - Bayesian Model Averaging

---

**创建日期**: 2026-04-16
**最后更新**: 2026-04-16
