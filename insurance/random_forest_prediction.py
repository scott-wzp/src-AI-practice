import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
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

# 划分训练集、验证集和测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# 训练随机森林模型
print("训练随机森林模型...")
rf_model = RandomForestClassifier(
    n_estimators=200,      # 增加树的数量
    max_depth=6,           # 增加树的深度
    min_samples_split=3,   # 降低分裂所需的最小样本数
    min_samples_leaf=1,    # 降低叶节点所需的最小样本数
    class_weight='balanced',  # 添加类别权重平衡
    random_state=42
)
rf_model.fit(X_train, y_train)

# 计算各数据集的准确率
train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
val_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

print("\n" + "="*50)
print("模型准确率评估")
print("="*50)
print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")
print("="*50 + "\n")

# 1. 交叉验证
print("\n1. 交叉验证结果：")
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 2. 在测试集上的性能
print("\n2. 测试集性能：")
y_pred = rf_model.predict(X_test)
print("\n混淆矩阵：")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('images/rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 打印分类报告
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 3. ROC曲线
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.savefig('images/rf_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\n3. 特征重要性：")
print(feature_importance.to_string(index=False))

# 可视化特征重要性
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('随机森林特征重要性（Top 10）')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.tight_layout()
plt.savefig('images/rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 保存验证结果到文件
with open('rf_validation_results.txt', 'w', encoding='utf-8') as f:
    f.write("随机森林模型验证结果\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. 各数据集准确率\n")
    f.write(f"训练集准确率: {train_accuracy:.4f}\n")
    f.write(f"验证集准确率: {val_accuracy:.4f}\n")
    f.write(f"测试集准确率: {test_accuracy:.4f}\n\n")
    
    f.write("2. 交叉验证结果\n")
    f.write(f"5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
    
    f.write("3. 测试集性能\n")
    f.write("混淆矩阵:\n")
    f.write(str(cm) + "\n\n")
    
    f.write("分类报告:\n")
    f.write(classification_report(y_test, y_pred) + "\n")
    
    f.write("4. ROC曲线 AUC值\n")
    f.write(f"AUC = {roc_auc:.4f}\n\n")
    
    f.write("5. 特征重要性\n")
    f.write(feature_importance.to_string(index=False))

print("\n验证结果已保存到 rf_validation_results.txt")
print("可视化结果已保存到 images 目录") 