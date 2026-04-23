import pandas as pd

# 读取文件
df = pd.read_csv('submission/submission_improved.csv')

# 删除 id 列中的双引号
df['id'] = df['id'].astype(str).str.replace('"', '').str.strip()

# 保存文件
df.to_csv('submission/submission_improved.csv', index=False, quoting=0)

print("已完成！")
print(f"前 10 行 id 列:")
print(df['id'].head(10).tolist())
