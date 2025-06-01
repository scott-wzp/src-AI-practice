import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存模型的目录
if not os.path.exists('models'):
    os.makedirs('models')

# 读取数据
df = pd.read_excel('policy_data.xlsx')

# 数据预处理
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

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 获取特征重要性（系数）
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.coef_[0]
})

# 按重要性绝对值排序
feature_importance['abs_importance'] = abs(feature_importance['importance'])
feature_importance = feature_importance.sort_values('abs_importance', ascending=False)

# 打印特征重要性
print("\n特征重要性（系数）：")
print(feature_importance[['feature', 'importance']].to_string(index=False))

# 可视化特征重要性
plt.figure(figsize=(12, 8))
# 创建颜色映射（注意：反转顺序）
colors = ['red' if x < 0 else 'green' for x in feature_importance['importance']][::-1]
# 绘制条形图（注意：反转顺序）
bars = plt.barh(range(len(feature_importance)), feature_importance['importance'][::-1], color=colors)
# 设置y轴刻度和标签（注意：反转顺序）
plt.yticks(range(len(feature_importance)), feature_importance['feature'][::-1])
# 添加零线
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
# 设置标题和标签
plt.title('特征重要性（逻辑回归系数）', pad=20)
plt.xlabel('系数值')
plt.ylabel('特征')
# 调整布局
plt.tight_layout()
# 保存图片
plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 计算并打印模型性能
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\n模型性能：")
print(f"训练集准确率: {train_score:.4f}")
print(f"测试集准确率: {test_score:.4f}")

# 保存特征重要性到CSV文件
feature_importance[['feature', 'importance']].to_csv('feature_importance.csv', index=False, encoding='utf-8-sig')

# 保存模型和预处理组件
print("\n保存模型和预处理组件...")
joblib.dump(model, 'models/renewal_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
# 保存特征名称
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
print("模型和预处理组件已保存到 models 目录") 