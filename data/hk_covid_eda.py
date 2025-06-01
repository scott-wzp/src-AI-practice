import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查文件是否存在
file_path = '香港各区疫情数据_20250322.xlsx'
if not os.path.exists(file_path):
    print(f"错误：找不到文件 {file_path}")
    exit(1)

try:
    # 读取数据
    print(f"正在读取文件：{file_path}")
    df = pd.read_excel(file_path)
    
    # 显示数据基本信息
    print("\n数据基本信息：")
    print(df.info())
    print("\n数据列名：")
    print(df.columns.tolist())
    print("\n数据前5行：")
    print(df.head())
    print("\n数据基本统计描述：")
    print(df.describe())

    # 创建图表
    plt.figure(figsize=(15, 10))

    # 1. 各区域累计确诊分布
    plt.subplot(2, 2, 1)
    sns.barplot(data=df, x='地区名称', y='累计确诊')
    plt.title('各区域累计确诊分布')
    plt.xticks(rotation=45)

    # 2. 各区域累计死亡分布
    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x='地区名称', y='累计死亡')
    plt.title('各区域累计死亡分布')
    plt.xticks(rotation=45)

    # 3. 累计确诊与累计死亡的散点图
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='累计确诊', y='累计死亡')
    plt.title('累计确诊与累计死亡关系')

    # 4. 各区域死亡率
    plt.subplot(2, 2, 4)
    df['死亡率'] = df['累计死亡'] / df['累计确诊'] * 100
    sns.barplot(data=df, x='地区名称', y='死亡率')
    plt.title('各区域死亡率')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('hk_covid_analysis.png')
    plt.close()

    # 热力图：地区-日期累计确诊数
    # 透视表，行是地区，列是日期，值是累计确诊
    heatmap_data = df.pivot_table(index='地区名称', columns='报告日期', values='累计确诊', aggfunc='max', fill_value=0)
    plt.figure(figsize=(18, 8))
    sns.heatmap(heatmap_data, cmap='Reds', linewidths=0.5)
    plt.title('各区每日累计确诊热力图')
    plt.xlabel('日期')
    plt.ylabel('地区名称')
    plt.tight_layout()
    plt.savefig('hk_covid_heatmap.png')
    plt.close()

    # 输出一些关键统计信息
    print("\n各区域死亡率：")
    print(df[['地区名称', '死亡率']].sort_values('死亡率', ascending=False))

    # 计算总体死亡率
    total_death_rate = df['累计死亡'].sum() / df['累计确诊'].sum() * 100
    print(f"\n总体死亡率: {total_death_rate:.2f}%")

    print("\n已生成分析图：hk_covid_analysis.png 和 热力图：hk_covid_heatmap.png")

except Exception as e:
    print(f"发生错误：{str(e)}") 