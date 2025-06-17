"""
Titanic_demo-
根据Tianic的数据集，分析获救人员的相关系数特征
Author: wzpym
Date: 2025/6/17
"""
import pandas as pd
import numpy as np

#获取数据
data = pd.read_csv("./titanic/train.csv")

#筛选特征值和目标值
x = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']  # 修改：直接使用Series而不是DataFrame

#数据处理：
#1)缺失值处理
# Convert Age to numeric, coercing errors to NaN
x['Age'] = pd.to_numeric(x['Age'], errors='coerce')
# Fill NaN values with mean
x['Age'].fillna(x['Age'].mean(), inplace=True)

#转换为字典形式的数据
x = x.to_dict(orient='records')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

#字典特征抽取
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz

transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#3)决策树预估器,使用决策树，需要设置参数，这里用信息熵，没有使用gini系数
estimator = DecisionTreeClassifier(criterion="entropy")
#训练模型
estimator.fit(x_train, y_train)

#4)模型评估
y_predict = estimator.predict(x_test)
print('y_predict:\n', y_predict)
print('直接比对真实值和预测值:\n', y_test == y_predict)
# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print('准确率为:\n', score)

#5)可视化决策树
export_graphviz(estimator, out_file="./titanic_tree.dot", feature_names=transfer.get_feature_names_out())
