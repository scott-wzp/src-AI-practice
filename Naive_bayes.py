"""
Naive_bayes-
朴素贝叶斯算法和拉普拉丝系数对新闻进行分类
Author: wzpym
Date: 2025/6/17
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier,export_graphviz

def naivebayes_news():
    #1)获取数据集
    news = fetch_20newsgroups(subset= 'all')
    #2)划分数据集
    x_train,x_test,y_train,y_test=train_test_split( news.data , news.target)
    #3)特征工程，文本特征抽取TF-IDF
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #4)朴素贝叶斯算法预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)
    print('y_train:\n',y_train)
    print('y_test:\n', y_test)
    #5)模型评估
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('直接比对真实值和预测值:\n', y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print('准确率为:\n', score)
    return None
def decision_iris():
    """用决策树对鸢尾花进行分类
       return:
    """
    #1)获取数据集
    iris = load_iris()
    #2)划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data , iris.target , random_state=22)
    #3)决策树预估器,使用决策树，需要设置参数，这里用信息熵，没有使用gini系数
    estimator  = DecisionTreeClassifier(criterion= "entropy")
    #训练模型
    estimator.fit(x_train,y_train)
    #4)模型评估
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('直接比对真实值和预测值:\n', y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print('准确率为:\n', score)
    #5)可视化决策树
    export_graphviz(estimator , out_file="./iris_tree.dot" , feature_names= iris.feature_names)

if __name__ == '__main__':
    #code1 朴素贝叶斯算法
    #naivebayes_news()
    #code2 信息熵算法使用
    decision_iris()