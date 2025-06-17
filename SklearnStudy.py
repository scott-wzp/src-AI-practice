"""
SklearnStudy-
机器学习，使用sklearn来获得数据集
Author: wzpym
Date: 2025/6/11
"""
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import jieba
import pandas as pd


def sklearn_dataset_use():
   #获取数据集
   iris = load_iris()
   print("iris数据集样式:\n",iris)
   print("查看数据集描述:\n",iris['DESCR'])
   print("查看特征值的名字:\n",iris.feature_names)
   print("查看特征值:\n",iris.data,iris.data.shape)
   #对数据集进行划分
   x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
   print("训练集的数量：\n",x_train.shape)
   return None

def dict_demo():
   """字典特征抽取"""
   data = [{'city':'成都','temperature':37},{'city':'上海','temperature':40},{'city':'北京','temperature':38}]
   #实例化转换器，调用方法fit_transform(传一个迭代器)
   #transfer = DictVectorizer()
   #参数sparse控制生成的矩阵是否为稀疏矩阵
   transfer = DictVectorizer(sparse=False)
   data__new = transfer.fit_transform(data)
   print('data_new:\n',data__new)
   #打印出特征名字
   print('特征名字:\n',transfer.feature_names_)

def count_demo():
   """文本特征抽取"""
   data =['life is short, i like like python','life is too long,i dislike python']
   #实例化转换器
   transfer = CountVectorizer()
   #调用转换器
   data_new = transfer.fit_transform(data)
   print('data_new:\n',data_new.toarray())

def count_chinese_demo():
   """文本特征抽取"""
   data =['我爱成都','成都是四川的省会城市']
   #实例化转换器
   transfer = CountVectorizer()
   #调用转换器
   data_new = transfer.fit_transform(data)
   print('data_new:\n',data_new.toarray())
   print('特征名字:\n',transfer.get_feature_names_out())

def cut_words(text):
   #此时产生一个生成器，然后需要将该生成器强转为list,然后对list里面的东西加上空格分隔
   #print(list(jieba.cut(text)))
   text = ' '.join(list(jieba.cut(text)))
   #print(' '.join(list(jieba.cut(text))))
   return text

def count_chinese_demo2():
   """中文特征提取2，将中文文本进行分词"""
   data=['SOTN控制器应实现对所控制设备网元的集中式控制。控制器应能够独立进行业务发放、保护恢复路由的计算等功能。',
         'SOTN控制器应部署在独立于OTN设备的服务器上，与OTN设备之间通过接口协议交互信令，实现资源信息上报、查询和配置下发等功能。']
   data_new = []
   for sent in data:
      data_new.append(cut_words(sent))
   print(data_new)
   #实现一个转换器类
   transfer = CountVectorizer(stop_words=['应','等'])
   #调用转换器
   data_final = transfer.fit_transform(data_new)
   print('data_final:\n', data_final.toarray())
   print('特征名字:\n',transfer.get_feature_names_out())

def count_di_idf_demo():
   """用tfidf的方式对文本进行特征抽取"""
   data=['SOTN控制器应实现对所控制设备网元的集中式控制。控制器应能够独立进行业务发放、保护恢复路由的计算等功能。',
         'SOTN控制器应部署在独立于OTN设备的服务器上，与OTN设备之间通过接口协议交互信令，实现资源信息上报、查询和配置下发等功能。']
   data_new = []
   for sent in data:
      data_new.append(cut_words(sent))
   print(data_new)
   #实现一个转换器类
   transfer = TfidfVectorizer(stop_words=['应','等'])
   #调用转换器
   data_final = transfer.fit_transform(data_new)
   print('data_final:\n', data_final.toarray())
   print('特征名字:\n',transfer.get_feature_names_out())

def min_max_scaler_demo():
   """归一化数据"""
   #获取数据
   data = pd.read_excel('./员工绩效表.xlsx')
   data = data.iloc[:5,:5]
   print('data:\n',data)
   #转换器类,可以设置归一化的范围，默认为0,1
   #transfer = MinMaxScaler()
   transfer = MinMaxScaler(feature_range=(2,3))
   #转换
   data_new = transfer.fit_transform(data)
   print('data_new:\n',data_new)

def standard_scaler_demo():
   """标准化数据"""
   # 获取数据
   data = pd.read_excel('./员工绩效表.xlsx')
   data = data.iloc[:5, :5]
   print('data:\n', data)
   # 转换器类,可以设置归一化的范围，默认为0,1
   # transfer = MinMaxScaler()
   transfer = StandardScaler()
   # 转换
   data_new = transfer.fit_transform(data)
   print('data_new:\n', data_new)

def variance_demo():
   """过滤低方差数据"""
   #1.获取数据
   data = pd.read_excel('./香港各区疫情数据_20250322.xlsx')
   r = pearsonr(data['累计康复'], data['新增死亡'])
   print('相关系数:\n', r)
   data = data.iloc[1:10,2:-2]
   print('data:\n',data)
   #2.构建转换器
   transfer = VarianceThreshold(threshold=0)
   #3.调用fit_transform
   data_new = transfer.fit_transform(data)
   # 将numpy数组转换回DataFrame，保持列名
   data_new = pd.DataFrame(data_new)
   print('data_new:\n',data_new,data_new.shape)

def pca_demo():
   """PCA降维，主要成分降维"""
   #1.准备数据
   data=[[2,8,4,5],[6,3,0,8],[5,4,9,1]]
   #实例化转换器类，将4个特征转换为2个特征;保留95%的特征
   transfer = PCA(n_components= 2)
   #transfer = PCA(n_components=0.95)
   #调用fit_transform
   data_new = transfer.fit_transform(data)
   print('PCA降维：\n',data_new)


if __name__=="__main__":

    #code1 dataset数据集的使用
    #sklearn_dataset_use()
    #code2 字典特征抽取
    #dict_demo()
    #code3 文本的特征抽取
    #count_demo()
    #code4 中文文本特征抽取
    count_chinese_demo()
    #code5 对文字进行拆分
    #cut_words("我爱成都，成都是天府之国")
    #code6 对中文字符串进行提取
    #count_chinese_demo2()
    #code7 对字符进行TFIDF特征抽取
    #count_di_idf_demo()
    #code8 对数据进行pandas读取
    #min_max_scaler_demo()
    #code9 进行标准化数据
    #standard_scaler_demo()
    #code10 低方差特征过滤
    #variance_demo()
    #code11 PCA降维
    #pca_demo()