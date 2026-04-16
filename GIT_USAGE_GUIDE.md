# Git 版本管理使用指南

本指南帮助你管理机器学习实验的 Git 版本，确保实验过程可追溯、结果可回退。

## 一、日常使用流程

### 每次实验完成后

```powershell
# 1. 进入项目目录
cd "d:\kaggle 词袋遇到\word2vec-nlp-tutorial"

# 2. 查看当前改动
git status

# 3. 添加所有改动
git add .

# 4. 提交改动 (写清楚本次实验内容)
git commit -m "feat: 完成情感分析模型训练，准确率 0.85"

# 5. 推送到 GitHub
git push
```

### 提交信息规范

**好的提交信息**:
- `feat: 添加 Word2Vec 词向量训练代码`
- `fix: 修复文本预处理中的标点符号处理 bug`
- `docs: 更新实验报告，添加结果分析`
- `result: 提交 Kaggle 结果，Public Score: 0.872`

**不好的提交信息**:
- `update`
- `test`
- `fix bug`
- `修改代码`

---

## 二、查看历史记录

### 查看提交历史

```powershell
# 简洁模式
git log --oneline

# 详细信息
git log

# 查看特定文件的修改历史
git log -- code\train_model.py

# 图形化查看
git log --graph --oneline --all
```

### 查看某次提交的详情

```powershell
# 查看某次提交的具体改动
git show <commit_id>

# 例如
git show 3157d3a
```

### 比较差异

```powershell
# 比较工作区和暂存区
git diff

# 比较当前版本和上次提交
git diff HEAD

# 比较两个版本之间
git diff commit_id1 commit_id2
```

---

## 三、版本回退 (实验效果变差时)

### 方法 1: 使用 git revert (推荐，安全)

`git revert` 会创建一个新的提交来撤销指定提交的改动，保留历史记录。

```powershell
# 回退某次提交
git revert <commit_id>

# 例如，回退最近的提交
git revert HEAD

# 回退多次提交
git revert commit_id1 commit_id2
```

**优点**: 
- 保留完整历史记录
- 可以安全地撤销已推送的提交
- 团队协作友好

### 方法 2: 使用 git reset (谨慎使用)

`git reset` 会直接删除提交记录。

```powershell
# 软回退：保留更改，仅撤销提交
git reset --soft <commit_id>

# 混合回退：保留文件更改，撤销提交和暂存
git reset --mixed <commit_id>

# 硬回退：完全恢复到指定版本 (危险!)
git reset --hard <commit_id>

# 例如，回退到上一个版本
git reset --hard HEAD~1
```

**警告**: 
- `--hard` 模式会永久删除更改
- 如果已经推送到远程，需要强制推送：`git push -f`
- 不要在团队协作中使用强制推送

### 方法 3: 恢复到历史版本并创建新分支

```powershell
# 基于历史版本创建新分支
git checkout -b backup_branch <commit_id>

# 之后可以比较或合并
git checkout main
git merge backup_branch
```

---

## 四、标签管理 (标记重要版本)

### 创建标签

```powershell
# 给当前版本打标签
git tag v1.0

# 给特定提交打标签
git tag v1.0 <commit_id>

# 带说明的标签
git tag -a v1.0 -m "第一个可用版本，准确率 0.85"
```

### 查看标签

```powershell
# 查看所有标签
git tag

# 查看标签详情
git show v1.0
```

### 推送标签到远程

```powershell
# 推送单个标签
git push origin v1.0

# 推送所有标签
git push origin --tags
```

### 删除标签

```powershell
# 删除本地标签
git tag -d v1.0

# 删除远程标签
git push origin --delete v1.0
```

---

## 五、分支管理

### 创建实验分支

```powershell
# 基于当前分支创建新分支
git branch experiment_lstm

# 创建并切换到新分支
git checkout -b experiment_lstm

# 查看分支列表
git branch

# 切换到其他分支
git checkout main
```

### 合并分支

```powershell
# 切换到主分支
git checkout main

# 合并实验分支
git merge experiment_lstm

# 删除已合并的分支
git branch -d experiment_lstm
```

---

## 六、常见问题处理

### 问题 1: 提交后发现有文件忘记添加

```powershell
# 添加忘记的文件
git add <forgotten_file>

# 修改上次提交
git commit --amend -m "正确的提交信息"

# 如果已经推送，需要强制推送
git push -f
```

### 问题 2: 提交信息写错了

```powershell
# 修改最近一次提交信息
git commit --amend -m "正确的提交信息"

# 如果已经推送
git push -f
```

### 问题 3: 想要撤销工作区的修改

```powershell
# 撤销某个文件的修改
git checkout -- <file_name>

# 撤销所有修改 (谨慎!)
git checkout -- .
```

### 问题 4: 想查看某个文件的历史版本

```powershell
# 查看文件历史
git log -- <file_name>

# 恢复文件到某个历史版本
git checkout <commit_id> -- <file_name>
```

---

## 七、实验版本管理最佳实践

### 1. 及时提交
- 每次实验完成后立即提交
- 不要等到最后一次一起提交
- 保持小步提交，便于回退

### 2. 提交信息清晰
- 说明做了什么改进
- 记录实验参数和结果
- 使用统一的提交格式

### 3. 重要版本打标签
- 效果好的版本打标签
- 提交到 Kaggle 前打标签
- 标签命名：`v1.0`, `v2.0`, `best_model`

### 4. 使用分支管理不同实验方向
- `main`: 主分支，稳定版本
- `experiment_xxx`: 实验分支，尝试新方法
- `backup_xxx`: 备份分支，保存重要版本

### 5. 定期推送到 GitHub
- 防止本地数据丢失
- 便于在不同设备上继续工作
- 保留完整实验历史

---

## 八、快速参考卡片

```powershell
# 日常使用
git status                    # 查看状态
git add .                     # 添加所有文件
git commit -m "说明"          # 提交
git push                      # 推送

# 查看历史
git log --oneline             # 查看提交历史
git show <id>                 # 查看详情

# 回退版本
git revert <id>               # 安全回退
git reset --hard <id>         # 强制回退 (谨慎!)

# 标签管理
git tag v1.0                  # 创建标签
git push origin --tags        # 推送标签

# 分支管理
git branch                    # 查看分支
git checkout -b new_branch    # 创建并切换分支
git merge branch_name         # 合并分支
```

---

## 九、恢复误操作

### 如果不小心 reset --hard 了

```powershell
# 查看操作历史
git reflog

# 找到 reset 之前的 commit id
# 恢复到那个版本
git reset --hard <commit_id_before_reset>
```

### 如果强制推送覆盖了别人的提交

```powershell
# 立即通知团队成员
# 使用 git revert 而不是 reset
# 或者协调后重新推送
```

---

**记住**: Git 是为了让你放心地做实验，效果不好可以随时回退!
