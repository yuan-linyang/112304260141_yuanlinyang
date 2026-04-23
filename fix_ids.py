import pandas as pd

df = pd.read_csv('submission/submission_improved.csv')
df['id'] = df['id'].astype(str).str.replace('"', '').str.strip()
df.to_csv('submission/submission_improved.csv', index=False, quoting=0)
print("已修正 ID 列的双引号")
print(f"前 5 行:\n{df.head()}")
