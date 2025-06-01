# hk_covid_eda.py 脚本逻辑说明

本脚本用于对《香港各区疫情数据_20250322.xlsx》文件进行探索性数据分析（EDA），并生成多种可视化图表，包括柱状图、散点图和热力图。

---

## 1. 导入依赖包
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
```
- 用于数据处理、可视化和文件操作。

## 2. 设置中文字体
```python
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```
- 保证图表中中文正常显示，负号正常显示。

## 3. 检查数据文件是否存在
```python
file_path = '香港各区疫情数据_20250322.xlsx'
if not os.path.exists(file_path):
    print(f"错误：找不到文件 {file_path}")
    exit(1)
```
- 如果数据文件不存在则直接退出。

## 4. 读取数据并输出基本信息
```python
try:
    print(f"正在读取文件：{file_path}")
    df = pd.read_excel(file_path)
    print("\n数据基本信息：")
    print(df.info())
    print("\n数据列名：")
    print(df.columns.tolist())
    print("\n数据前5行：")
    print(df.head())
    print("\n数据基本统计描述：")
    print(df.describe())
```
- 读取Excel数据，输出数据结构、列名、前5行和统计描述，便于了解数据内容。

## 5. 生成多种可视化图表
### 5.1 各区域累计确诊分布
```python
plt.subplot(2, 2, 1)
sns.barplot(data=df, x='地区名称', y='累计确诊')
plt.title('各区域累计确诊分布')
plt.xticks(rotation=45)
```
- 展示每个地区的累计确诊病例数。

### 5.2 各区域累计死亡分布
```python
plt.subplot(2, 2, 2)
sns.barplot(data=df, x='地区名称', y='累计死亡')
plt.title('各区域累计死亡分布')
plt.xticks(rotation=45)
```
- 展示每个地区的累计死亡病例数。

### 5.3 累计确诊与累计死亡的散点图
```python
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='累计确诊', y='累计死亡')
plt.title('累计确诊与累计死亡关系')
```
- 展示累计确诊与累计死亡之间的关系。

### 5.4 各区域死亡率
```python
plt.subplot(2, 2, 4)
df['死亡率'] = df['累计死亡'] / df['累计确诊'] * 100
sns.barplot(data=df, x='地区名称', y='死亡率')
plt.title('各区域死亡率')
plt.xticks(rotation=45)
```
- 计算并展示每个地区的死亡率。

### 5.5 保存上述图表
```python
plt.tight_layout()
plt.savefig('hk_covid_analysis.png')
plt.close()
```
- 保存所有子图到一张图片。

### 5.6 地区-日期累计确诊热力图
```python
heatmap_data = df.pivot_table(index='地区名称', columns='报告日期', values='累计确诊', aggfunc='max', fill_value=0)
plt.figure(figsize=(18, 8))
sns.heatmap(heatmap_data, cmap='Reds', linewidths=0.5)
plt.title('各区每日累计确诊热力图')
plt.xlabel('日期')
plt.ylabel('地区名称')
plt.tight_layout()
plt.savefig('hk_covid_heatmap.png')
plt.close()
```
- 以地区为行、日期为列，累计确诊为值，生成热力图，直观展示疫情随时间和地区的分布。

## 6. 输出关键统计信息
```python
print("\n各区域死亡率：")
print(df[['地区名称', '死亡率']].sort_values('死亡率', ascending=False))

# 计算总体死亡率
print(f"\n总体死亡率: {total_death_rate:.2f}%")
```
- 输出各地区死亡率和总体死亡率。

## 7. 错误处理
```python
except Exception as e:
    print(f"发生错误：{str(e)}")
```
- 捕获并输出所有异常，便于调试。

---

**输出文件：**
- `hk_covid_analysis.png`：四个子图的综合分析图
- `hk_covid_heatmap.png`：地区-日期累计确诊热力图 