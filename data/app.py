from flask import Flask, render_template, jsonify, send_from_directory
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

app = Flask(__name__)

# 读取数据
print("正在读取数据文件...")
df = pd.read_excel('香港各区疫情数据_20250322.xlsx')
print(f"数据读取完成，共 {len(df)} 行")
# 确保报告日期列为datetime类型
df['报告日期'] = pd.to_datetime(df['报告日期'])
print("日期格式转换完成")

# 自定义 JSON 编码器处理 NaN 值
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and np.isnan(obj):
            return 0
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

@app.route('/')
def index():
    print("访问主页")
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/daily_cases')
def daily_cases():
    print("请求每日病例数据")
    # 按日期统计每日新增和累计确诊
    daily_stats = df.groupby('报告日期').agg({
        '新增确诊': 'sum',
        '累计确诊': 'max'
    }).reset_index()
    
    # 确保日期格式正确
    dates = daily_stats['报告日期'].dt.strftime('%Y-%m-%d').tolist()
    
    # 将 NaN 值替换为 0
    new_cases = daily_stats['新增确诊'].fillna(0).astype(int).tolist()
    total_cases = daily_stats['累计确诊'].fillna(0).astype(int).tolist()
    
    response_data = {
        'dates': dates,
        'new_cases': new_cases,
        'total_cases': total_cases
    }
    print(f"返回每日病例数据: {len(dates)} 条记录")
    return jsonify(response_data)

@app.route('/api/district_stats')
def district_stats():
    print("请求地区统计数据")
    # 获取最新的地区统计数据
    latest_date = df['报告日期'].max()
    latest_data = df[df['报告日期'] == latest_date]
    print(f"最新数据日期: {latest_date}")
    
    # 计算每个地区的风险等级（基于新增确诊数）
    risk_levels = []
    for district in latest_data['地区名称']:
        district_data = df[df['地区名称'] == district]
        recent_cases = district_data['新增确诊'].tail(7).mean()  # 最近7天平均新增
        if pd.isna(recent_cases) or recent_cases > 100:
            risk_levels.append('high')
        elif pd.isna(recent_cases) or recent_cases > 50:
            risk_levels.append('medium')
        else:
            risk_levels.append('low')
    
    # 将 NaN 值替换为 0
    cases = latest_data['累计确诊'].fillna(0).astype(int).tolist()
    deaths = latest_data['累计死亡'].fillna(0).astype(int).tolist()
    rates = (latest_data['累计死亡'].fillna(0) / latest_data['累计确诊'].fillna(1) * 100).round(2).fillna(0).tolist()
    
    response_data = {
        'districts': latest_data['地区名称'].tolist(),
        'cases': cases,
        'deaths': deaths,
        'rates': rates,
        'risk_levels': risk_levels
    }
    print(f"返回地区统计数据: {len(latest_data)} 个地区")
    return jsonify(response_data)

@app.route('/api/trend_analysis')
def trend_analysis():
    print("请求趋势分析数据")
    # 计算每日增长率
    daily_stats = df.groupby('报告日期').agg({
        '新增确诊': 'sum',
        '累计确诊': 'max'
    }).reset_index()
    
    # 计算增长率，将 NaN 值替换为 0
    daily_stats['growth_rate'] = daily_stats['新增确诊'].fillna(0).pct_change().fillna(0) * 100
    
    # 确保日期格式正确
    dates = daily_stats['报告日期'].dt.strftime('%Y-%m-%d').tolist()
    
    # 将 NaN 值替换为 0
    growth_rates = daily_stats['growth_rate'].round(2).fillna(0).tolist()
    new_cases = daily_stats['新增确诊'].fillna(0).astype(int).tolist()
    
    response_data = {
        'dates': dates,
        'growth_rates': growth_rates,
        'new_cases': new_cases
    }
    print(f"返回趋势分析数据: {len(dates)} 条记录")
    return jsonify(response_data)

if __name__ == '__main__':
    print("启动 Flask 应用...")
    app.run(debug=True) 