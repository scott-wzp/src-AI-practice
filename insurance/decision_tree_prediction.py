import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图片的目录
if not os.path.exists('images'):
    os.makedirs('images')

# 读取数据
print("读取数据...")
df = pd.read_excel('policy_data.xlsx')

# 数据预处理
print("数据预处理...")
# 1. 将目标变量转换为数值型
le = LabelEncoder()
df['renewal'] = le.fit_transform(df['renewal'])

# 2. 选择特征并进行编码
categorical_features = ['gender', 'income_level', 'education_level', 
                       'occupation', 'marital_status', 'policy_type', 
                       'policy_term', 'claim_history']
numerical_features = ['age', 'family_members', 'premium_amount']

# 对分类特征进行编码
X_categorical = pd.get_dummies(df[categorical_features], drop_first=True)
X_numerical = df[numerical_features]

# 合并特征
X = pd.concat([X_numerical, X_categorical], axis=1)
y = df['renewal']

# 训练决策树模型
print("训练决策树模型...")
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X, y)

# 打印决策树结构
print("\n决策树结构：")
tree_text = export_text(dt_model, feature_names=X.columns.tolist())
print(tree_text)

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(dt_model, 
          feature_names=X.columns.tolist(),
          class_names=['不续保', '续保'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('决策树结构可视化', pad=20)
plt.savefig('images/decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# 计算并打印模型性能
train_score = dt_model.score(X, y)
print(f"\n模型性能：")
print(f"训练集准确率: {train_score:.4f}")

# 获取特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
})

# 按重要性排序
feature_importance = feature_importance.sort_values('importance', ascending=False)

# 打印特征重要性
print("\n特征重要性：")
print(feature_importance.to_string(index=False))

# 可视化特征重要性
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('决策树特征重要性')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.tight_layout()
plt.savefig('images/decision_tree_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n可视化结果已保存到 images 目录") 