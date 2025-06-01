import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
import seaborn as sns
import os

# 创建保存图片的目录
if not os.path.exists('images'):
    os.makedirs('images')

# 读取数据
df = pd.read_excel('policy_data.xlsx')

# 基本统计信息
print("基本统计信息：")
print(df.describe())

# 检查缺失值
print("\n缺失值检查：")
print(df.isnull().sum())

# 数据分布
print("\n数据分布：")
for column in df.select_dtypes(include=['object']).columns:
    print(f"\n{column} 的分布：")
    print(df[column].value_counts())

# 可视化数据分布
plt.figure(figsize=(10, 6))
sns.countplot(x='gender', data=df)
plt.title('性别分布')
plt.savefig('images/gender_distribution.png')
plt.close()

# 可视化年龄分布
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('年龄分布')
plt.savefig('images/age_distribution.png')
plt.close()

# 可视化收入水平分布
plt.figure(figsize=(10, 6))
sns.countplot(x='income_level', data=df)
plt.title('收入水平分布')
plt.savefig('images/income_distribution.png')
plt.close()

# 可视化职业分布
plt.figure(figsize=(10, 6))
sns.countplot(x='occupation', data=df)
plt.title('职业分布')
plt.savefig('images/occupation_distribution.png')
plt.close()

# 可视化保费金额分布
plt.figure(figsize=(10, 6))
sns.histplot(df['premium_amount'], bins=30, kde=True)
plt.title('保费金额分布')
plt.savefig('images/premium_distribution.png')
plt.close()

# 可视化理赔历史分布
plt.figure(figsize=(10, 6))
sns.countplot(x='claim_history', data=df)
plt.title('理赔历史分布')
plt.savefig('images/claim_history_distribution.png')
plt.close()

# 可视化婚姻状况分布
plt.figure(figsize=(10, 6))
sns.countplot(x='marital_status', data=df)
plt.title('婚姻状况分布')
plt.savefig('images/marital_status_distribution.png')
plt.close()

# 可视化教育水平分布
plt.figure(figsize=(10, 6))
sns.countplot(x='education_level', data=df)
plt.title('教育水平分布')
plt.savefig('images/education_distribution.png')
plt.close()

# 可视化地区分布
plt.figure(figsize=(10, 6))
sns.countplot(x='insurance_region', data=df)
plt.title('地区分布')
plt.savefig('images/region_distribution.png')
plt.close()

print("\n所有图片已保存到 images 目录下") 