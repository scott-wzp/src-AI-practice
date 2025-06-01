import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 读取数据
print("读取数据...")
df = pd.read_excel('policy_data.xlsx')
test_df = pd.read_excel('policy_test.xlsx')

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

# 对训练数据进行编码
X_categorical = pd.get_dummies(df[categorical_features], drop_first=True)
X_numerical = df[numerical_features]
X = pd.concat([X_numerical, X_categorical], axis=1)
y = df['renewal']

# 训练决策树模型
print("训练决策树模型...")
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X, y)

# 对测试数据进行编码
print("处理测试数据...")
test_categorical = pd.get_dummies(test_df[categorical_features], drop_first=True)
test_numerical = test_df[numerical_features]
X_test = pd.concat([test_numerical, test_categorical], axis=1)

# 确保测试数据的特征与训练数据一致
for col in X.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X.columns]

# 进行预测
print("进行预测...")
predictions = dt_model.predict(X_test)
probabilities = dt_model.predict_proba(X_test)

# 将预测结果添加到原始数据中
test_df['predicted_renewal'] = le.inverse_transform(predictions)
test_df['renewal_probability'] = probabilities[:, 1]  # 续保的概率

# 保存预测结果
print("保存预测结果...")
test_df.to_excel('decision_tree_predictions.xlsx', index=False)

# 打印预测结果统计
print("\n预测结果统计：")
print(f"总样本数: {len(test_df)}")
print(f"预测续保数: {sum(predictions == 1)}")
print(f"预测不续保数: {sum(predictions == 0)}")

# 分析预测结果
print("\n预测结果分析：")
print("\n按年龄段的预测结果：")
test_df['age_group'] = pd.cut(test_df['age'], 
                             bins=[0, 30, 45, 60, 100],
                             labels=['青年', '中年', '中老年', '老年'])
age_group_stats = test_df.groupby('age_group')['predicted_renewal'].value_counts().unstack()
print(age_group_stats)

print("\n按婚姻状况的预测结果：")
marital_stats = test_df.groupby('marital_status')['predicted_renewal'].value_counts().unstack()
print(marital_stats)

print("\n按职业的预测结果：")
occupation_stats = test_df.groupby('occupation')['predicted_renewal'].value_counts().unstack()
print(occupation_stats)

# 保存详细分析结果
with open('prediction_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("决策树模型预测分析报告\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. 总体预测结果\n")
    f.write(f"总样本数: {len(test_df)}\n")
    f.write(f"预测续保数: {sum(predictions == 1)}\n")
    f.write(f"预测不续保数: {sum(predictions == 0)}\n\n")
    
    f.write("2. 按年龄段的预测结果\n")
    f.write(age_group_stats.to_string() + "\n\n")
    
    f.write("3. 按婚姻状况的预测结果\n")
    f.write(marital_stats.to_string() + "\n\n")
    
    f.write("4. 按职业的预测结果\n")
    f.write(occupation_stats.to_string() + "\n")

print("\n预测结果已保存到 decision_tree_predictions.xlsx")
print("详细分析报告已保存到 prediction_analysis.txt") 