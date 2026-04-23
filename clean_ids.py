import pandas as pd

# 处理 submission_test.csv
print("处理 submission_test.csv...")
df = pd.read_csv('submission/submission_test.csv')
df['id'] = df['id'].astype(str).str.replace('"', '').str.strip()
df.to_csv('submission/submission_test.csv', index=False, quoting=0)
print(f"完成！前 5 行 id: {df['id'].head().tolist()}")

# 处理 submission_word2vec_mean_lr.csv
print("\n处理 submission_word2vec_mean_lr.csv...")
df = pd.read_csv('submission/submission_word2vec_mean_lr.csv')
df['id'] = df['id'].astype(str).str.replace('"', '').str.strip()
df.to_csv('submission/submission_word2vec_mean_lr.csv', index=False, quoting=0)
print(f"完成！前 5 行 id: {df['id'].head().tolist()}")

print("\n所有文件处理完成!")
