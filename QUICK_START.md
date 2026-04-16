# 🎓 机器学习实验 GitHub 管理 - 快速开始

## ✅ 已完成的配置

### 1. Git 环境配置
- ✅ Git 已安装并配置
- ✅ 用户名：yuan-linyang
- ✅ 邮箱：yuan-linyang@users.noreply.github.com

### 2. 项目结构已创建
```
word2vec-nlp-tutorial/
├─ code/              # 存放源代码
├─ data/              # 存放数据文件
├─ report/            # 存放实验报告
├─ results/           # 存放实验结果
├─ images/            # 存放截图
├─ submission/        # 存放提交文件
├─ README.md          # 实验说明 (含你的姓名学号)
├─ GIT_USAGE_GUIDE.md # Git 使用指南
└─ .gitignore         # Git 忽略文件配置
```

### 3. GitHub 仓库已连接
- ✅ 仓库地址：https://github.com/yuan-linyang/112304260141_yuanlinyang
- ✅ 初始版本已提交并推送
- ✅ 当前分支：main

---

## 🚀 后续使用 (每次实验后)

### 简单三步推送代码到 GitHub

```powershell
# 1. 添加改动
git add .

# 2. 提交改动 (写清楚做了什么)
git commit -m "feat: 完成情感分析模型，准确率 0.85"

# 3. 推送到 GitHub
git push
```

### 示例场景

#### 场景 1: 完成代码编写
```powershell
git add .
git commit -m "feat: 完成 Word2Vec 词向量训练代码"
git push
```

#### 场景 2: 更新实验报告
```powershell
git add .
git commit -m "docs: 更新实验报告，添加结果分析"
git push
```

#### 场景 3: 提交 Kaggle 结果
```powershell
git add .
git commit -m "result: Kaggle 提交，Public Score: 0.872"
git push
```

---

## 📋 实验内容检查清单

每次实验后，确保:

- [ ] 代码已放入 `code/` 目录
- [ ] 实验报告已放入 `report/` 目录
- [ ] 结果截图已放入 `images/` 目录
- [ ] 提交文件已放入 `submission/` 目录
- [ ] README.md 已更新 (Kaggle 分数等)
- [ ] 已提交并推送到 GitHub

---

## 🔄 如果实验效果变差

### 查看历史版本
```powershell
git log --oneline
```

### 回退到之前的版本
```powershell
# 方法 1: 安全回退 (推荐)
git revert <commit_id>
git push

# 方法 2: 强制回退 (谨慎使用)
git reset --hard <commit_id>
git push -f
```

**详细说明请查看**: [GIT_USAGE_GUIDE.md](./GIT_USAGE_GUIDE.md)

---

## 📝 提交信息规范

**好的提交信息**:
- `feat: 添加 LSTM 模型训练代码`
- `fix: 修复文本预处理 bug`
- `docs: 更新实验报告`
- `result: Kaggle 分数提升至 0.872`

**不好的提交信息**:
- ❌ `update`
- ❌ `test`
- ❌ `修改`

---

## 🔧 常用 Git 命令

```powershell
# 查看状态
git status

# 查看历史
git log --oneline

# 查看改动
git diff

# 撤销工作区改动
git checkout -- <文件名>

# 查看某个文件历史
git log -- <文件名>
```

---

## 📚 详细文档

- **Git 使用完整指南**: [GIT_USAGE_GUIDE.md](./GIT_USAGE_GUIDE.md)
- **实验报告模板**: [readme_机器学习实验 2 模板.md](./readme_机器学习实验 2 模板.md)
- **GitHub 仓库**: https://github.com/yuan-linyang/112304260141_yuanlinyang

---

## ⚠️ 注意事项

1. **不要上传敏感信息**: API 密钥、密码等已配置在 `.gitignore` 中
2. **不要上传大文件**: 数据集文件 (>100MB) 尽量避免上传
3. **及时提交**: 每次实验后立即提交，不要累积
4. **写清楚提交信息**: 便于日后查找和回退
5. **保留实验过程**: 不要只保留最终结果

---

## 🎯 目的

使用 GitHub 管理实验是为了:
- ✅ 实验过程有记录
- ✅ 每次修改可追踪
- ✅ 代码和报告不丢失
- ✅ 结果变差时可以回退
- ✅ 实验过程更规范

---

**祝你实验顺利!** 🎉
