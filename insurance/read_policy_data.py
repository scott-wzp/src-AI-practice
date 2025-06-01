import pandas as pd

# 设置显示选项，展示所有列
pd.set_option('display.max_columns', None)

# 读取 Excel 文件的前5行
df = pd.read_excel('policy_data.xlsx')
print(df.head(5)) 