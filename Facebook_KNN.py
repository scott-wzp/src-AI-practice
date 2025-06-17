"""
Facebook_KNN-

Author: wzpym
Date: 2025/6/16
"""
import pandas as pd
# 获取数据
data = pd.read_csv("./FaceBook_train.csv")
# 2.基本的数据处理
# 1）缩小数据范围
data = data.query("x<2.5 & x>2 & y<1.5 & y>1.0")
# 2)处理时间特征
time_value = pd.to_datetime(data["time"], unit="s")
date = pd.DatetimeIndex(time_value)
data["day"] = date.day
data['weekday'] = date.weekday
data['hour'] = date.hour
# 3)过滤签到次数少的地点
place_count = data.groupby('place_id').count()['row_id']
# place_count[place_count > 3]
data_final = data[data["place_id"].isin(place_count[place_count > 3].index.values)]
# 筛选特征值和目标值
x = data_final[["x", "y", "accuracy", "day", "weekday", "hour"]]
y = data_final["place_id"]
# 数据集划分
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 3 特征工程，标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
# 有了训练集，还需要有测试集，此时的测试集需要用和训练集一样的算法，因此，这里就直接调用transform方法，而训练数据的fit方法
# 沿用以前的fit_transform的方法
x_test = transfer.transform(x_test)
# 4 KNN算法预估器
estimator = KNeighborsClassifier()
# 加入网管搜索和交叉验证
# 参数准备：
param_dict = {"n_neighbors": [3, 5, 7, 9]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
estimator.fit(x_train, y_train)
# 5 模型评估
# 方法1：直接比对真实值和预测值
y_predict = estimator.predict(x_test)
print('y_predict:\n', y_predict)
print('直接比对真实值和预测值:\n', y_test == y_predict)
# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print('准确率为:\n', score)
# 最佳参数best_params_
print('最佳参数：\n', estimator.best_params_)
# 最佳结果best_score_
print('最佳结果：\n', estimator.best_score_)
# 最佳估计器best_estimator_
print('最佳估计器：\n', estimator.best_estimator_)
# 交叉验证结果:cv_results_
print('交叉验证结果：\n', estimator.cv_results_)

