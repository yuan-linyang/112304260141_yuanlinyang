# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：袁林扬
- **学号**：112304260141
- **班级**：

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/c/word2vec-nlp-tutorial
- **提交日期**：2026-04-16

- **GitHub 仓库地址**：https://github.com/yuan-linyang/112304260141_yuanlinyang
- **GitHub README 地址**：https://github.com/yuan-linyang/112304260141_yuanlinyang/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号",否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果:

- **Public Score**:
- **Private Score**(如有):
- **排名**(如能看到可填写):

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle 截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例:2023123456_张三_kaggle_score.png

---

## 6. 实验方法说明

### (1)文本预处理
请说明你对文本做了哪些处理，例如:
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法**:  
请在这里填写。

---

### (2)Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如:
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到 (平均、加权平均、池化等)

**我的做法**:  
请在这里填写。

---

### (3)分类模型
请说明你使用了什么分类模型，例如:
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法**:  
请在这里填写。

---

## 7. 实验流程
请简要说明你的实验流程。

示例:
1. 读取训练集和测试集  
2. 对文本进行预处理  
3. 训练或加载 Word2Vec 模型  
4. 将每条文本表示为句向量  
5. 用训练集训练分类器  
6. 在测试集上预测结果  
7. 生成 submission 文件并提交 Kaggle  

**我的实验流程**:  
请在这里填写。

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

**我的项目结构**:
```text
word2vec-nlp-tutorial/
├─ data/              # 存放数据文件 (训练集、测试集)
├─ code/              # 存放源代码
├─ report/            # 存放实验报告
├─ results/           # 存放实验结果、日志
├─ images/            # 存放 README 中使用的图片 (如 Kaggle 截图)
├─ submission/        # 存放提交到 Kaggle 的文件
├─ labeledTrainData.tsv.zip   # 训练数据
├─ testData.tsv.zip           # 测试数据
├─ unlabeledTrainData.tsv.zip # 无标签训练数据
├─ sampleSubmission.csv       # 提交样例
└─ README.md          # 仓库说明文档
```

---

## 9. Git 版本管理说明

### 常用 Git 命令

#### 查看历史提交记录
```bash
git log --oneline
```

#### 查看某个版本的详细内容
```bash
git show <commit_id>
```

#### 回退到历史版本
```bash
# 软回退 (保留更改，仅撤销提交)
git reset --soft <commit_id>

# 硬回退 (完全恢复到指定版本)
git reset --hard <commit_id>

# 创建新提交回退更改 (推荐)
git revert <commit_id>
```

#### 查看文件历史
```bash
git log -- <file_path>
```

### 实验版本管理建议
1. 每次实验完成后立即提交
2. 提交信息写清楚本次实验内容
3. 效果变差时使用 `git revert` 或 `git reset --hard` 恢复
4. 重要版本使用 tag 标记: `git tag v1.0`

---

## 10. 后续更新说明

每次实验后，将按以下步骤更新仓库:

1. **更新代码**: 将最新代码放入 `code/` 目录
2. **更新报告**: 将实验报告放入 `report/` 目录
3. **保存结果**: 将结果截图、表格放入 `results/` 和 `images/` 目录
4. **提交修改**:
   ```bash
   git add .
   git commit -m "描述本次实验的改进和内容"
   git push
   ```

---

**作者**: 袁林扬  
**学号**: 112304260141  
**创建日期**: 2026-04-16
