"""
IrisAnalyse-
对sklearn中自带的数据集进行分析，首先是获取数据，然后划分数据集，再特征工程：标准化后，再
KNN算法预估，最后模型评估
Author: wzpym
Date: 2025/6/16
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  GridSearchCV

def knn_iris():
    #1 获取数据集
    iris = load_iris()
    #2 划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data , iris.target ,random_state= 6)
    #3 特征工程，标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    #有了训练集，还需要有测试集，此时的测试集需要用和训练集一样的算法，因此，这里就直接调用transform方法，而训练数据的fit方法
    #沿用以前的fit_transform的方法
    x_test = transfer.transform(x_test)
    #4 KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)
    #5 模型评估
    #方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print('y_predict:\n' , y_predict)
    print('直接比对真实值和预测值:\n',y_test == y_predict)
    #方法2：计算准确率
    score = estimator.score(x_test , y_test)
    print('准确率为:\n',score)
    return None

def knn_iris_gscv():
    #1 获取数据集
    iris = load_iris()
    #2 划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data , iris.target ,random_state= 6)
    #3 特征工程，标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    #有了训练集，还需要有测试集，此时的测试集需要用和训练集一样的算法，因此，这里就直接调用transform方法，而训练数据的fit方法
    #沿用以前的fit_transform的方法
    x_test = transfer.transform(x_test)
    #4 KNN算法预估器
    estimator = KNeighborsClassifier()
    # 加入网管搜索和交叉验证
    # 参数准备：
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)
    # 5 模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print('y_predict:\n' , y_predict)
    print('直接比对真实值和预测值:\n',y_test == y_predict)
    #方法2：计算准确率
    score = estimator.score(x_test , y_test)
    print('准确率为:\n',score)

    #最佳参数best_params_
    print('最佳参数：\n',estimator.best_params_)
    #最佳结果best_score_
    print('最佳结果：\n',estimator.best_score_)
    #最佳估计器best_estimator_
    print('最佳估计器：\n', estimator.best_estimator_)
    #交叉验证结果:cv_results_
    print('交叉验证结果：\n',estimator.cv_results_)
    return None

if __name__ == "__main__":
    #KNN算法分析iris数据
    #knn_iris()
    # KNN算法分析iris数据,进行交叉验证和网格搜素
    knn_iris_gscv()