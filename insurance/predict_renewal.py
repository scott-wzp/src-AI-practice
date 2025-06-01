import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# 加载保存的模型和预处理组件
print("加载模型和预处理组件...")
model = joblib.load('models/renewal_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le = joblib.load('models/label_encoder.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# 读取测试数据
print("\n读取测试数据...")
test_df = pd.read_excel('policy_test.xlsx')

# 数据预处理
print("数据预处理...")
# 选择特征并进行编码
categorical_features = ['gender', 'income_level', 'education_level', 
                       'occupation', 'marital_status', 'policy_type', 
                       'policy_term', 'claim_history']
numerical_features = ['age', 'family_members', 'premium_amount']

# 对分类特征进行编码
X_categorical = pd.get_dummies(test_df[categorical_features], drop_first=True)
X_numerical = test_df[numerical_features]

# 合并特征
X = pd.concat([X_numerical, X_categorical], axis=1)

# 确保测试数据的特征与训练数据一致
for col in feature_names:
    if col not in X.columns:
        X[col] = 0
X = X[feature_names]

# 数据标准化
X_scaled = scaler.transform(X)

# 进行预测
print("进行预测...")
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)

# 将预测结果添加到原始数据中
test_df['predicted_renewal'] = le.inverse_transform(predictions)
test_df['renewal_probability'] = probabilities[:, 1]  # 续保的概率

# 保存预测结果
print("保存预测结果...")
test_df.to_excel('prediction_results.xlsx', index=False)

# 打印预测结果统计
print("\n预测结果统计：")
print(f"总样本数: {len(test_df)}")
print(f"预测续保数: {sum(predictions == 1)}")
print(f"预测不续保数: {sum(predictions == 0)}")
print("\n预测结果已保存到 prediction_results.xlsx") 