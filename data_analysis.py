import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_csv('processed_houses.csv')

# 数据基本信息
print("数据基本信息：")
print(f"数据量：{df.shape[0]}行，{df.shape[1]}列")
print("\n数据类型：")
print(df.dtypes)

# 数据统计描述
print("\n数据统计描述：")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe())

# 检查缺失值
print("\n缺失值统计：")
print(df.isnull().sum())

# 数据分析函数
def analyze_data(df):
    # 1. 价格分析
    plt.figure(figsize=(12, 8))
    
    # 售价分布
    plt.subplot(2, 2, 1)
    sns.histplot(df['售价'], kde=True)
    plt.title('二手房售价分布')
    plt.xlabel('售价(万元)')
    
    # 单价分布
    plt.subplot(2, 2, 2)
    sns.histplot(df['单价'], kde=True)
    plt.title('二手房单价分布')
    plt.xlabel('单价(元/平方米)')
    
    # 面积与价格关系
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='面积数值', y='售价', data=df)
    plt.title('面积与售价关系')
    plt.xlabel('面积(平方米)')
    plt.ylabel('售价(万元)')
    
    # 房间数与价格关系
    plt.subplot(2, 2, 4)
    sns.boxplot(x='房间数', y='售价', data=df)
    plt.title('房间数与售价关系')
    plt.xlabel('房间数')
    plt.ylabel('售价(万元)')
    
    plt.tight_layout()
    plt.savefig('price_analysis.png')
    
    # 2. 区域分析
    plt.figure(figsize=(14, 10))
    
    # 不同区域房源数量
    plt.subplot(2, 2, 1)
    region_count = df['区域'].value_counts().sort_values(ascending=False)
    sns.barplot(x=region_count.index[:15], y=region_count.values[:15])
    plt.title('各区域房源数量(Top 15)')
    plt.xticks(rotation=45)
    
    # 不同区域平均价格
    plt.subplot(2, 2, 2)
    region_price = df.groupby('区域')['售价'].mean().sort_values(ascending=False)
    sns.barplot(x=region_price.index[:15], y=region_price.values[:15])
    plt.title('各区域平均售价(Top 15)')
    plt.xticks(rotation=45)
    plt.ylabel('平均售价(万元)')
    
    # 不同区域单价
    plt.subplot(2, 2, 3)
    region_unit_price = df.groupby('区域')['单价'].mean().sort_values(ascending=False)
    sns.barplot(x=region_unit_price.index[:15], y=region_unit_price.values[:15])
    plt.title('各区域平均单价(Top 15)')
    plt.xticks(rotation=45)
    plt.ylabel('平均单价(元/平方米)')
    
    # 不同区域面积
    plt.subplot(2, 2, 4)
    region_area = df.groupby('区域')['面积数值'].mean().sort_values(ascending=False)
    sns.barplot(x=region_area.index[:15], y=region_area.values[:15])
    plt.title('各区域平均面积(Top 15)')
    plt.xticks(rotation=45)
    plt.ylabel('平均面积(平方米)')
    
    plt.tight_layout()
    plt.savefig('region_analysis.png')
    
    # 3. 户型分析
    plt.figure(figsize=(12, 8))
    
    # 户型分布
    plt.subplot(2, 2, 1)
    house_type_count = df['户型'].value_counts().sort_values(ascending=False)
    sns.barplot(x=house_type_count.index[:10], y=house_type_count.values[:10])
    plt.title('户型分布(Top 10)')
    plt.xticks(rotation=45)
    
    # 不同户型价格
    plt.subplot(2, 2, 2)
    house_type_price = df.groupby('户型')['售价'].mean().sort_values(ascending=False)
    sns.barplot(x=house_type_price.index[:10], y=house_type_price.values[:10])
    plt.title('不同户型平均售价(Top 10)')
    plt.xticks(rotation=45)
    plt.ylabel('平均售价(万元)')
    
    # 朝向分布
    plt.subplot(2, 2, 3)
    orientation_count = df['朝向'].value_counts().sort_values(ascending=False)
    sns.barplot(x=orientation_count.index[:10], y=orientation_count.values[:10])
    plt.title('朝向分布(Top 10)')
    plt.xticks(rotation=45)
    
    # 装修情况分布
    plt.subplot(2, 2, 4)
    decoration_count = df['装修'].value_counts()
    sns.barplot(x=decoration_count.index, y=decoration_count.values)
    plt.title('装修情况分布')
    
    plt.tight_layout()
    plt.savefig('house_type_analysis.png')
    
    # 4. 楼层分析
    plt.figure(figsize=(12, 8))
    
    # 楼层类型分布
    plt.subplot(2, 2, 1)
    floor_type_count = df['楼层类型'].value_counts()
    sns.barplot(x=floor_type_count.index, y=floor_type_count.values)
    plt.title('楼层类型分布')
    
    # 不同楼层类型价格
    plt.subplot(2, 2, 2)
    floor_type_price = df.groupby('楼层类型')['售价'].mean()
    sns.barplot(x=floor_type_price.index, y=floor_type_price.values)
    plt.title('不同楼层类型平均售价')
    plt.ylabel('平均售价(万元)')
    
    # 建筑结构分布
    plt.subplot(2, 2, 3)
    structure_count = df['建筑结构'].value_counts()
    sns.barplot(x=structure_count.index, y=structure_count.values)
    plt.title('建筑结构分布')
    plt.xticks(rotation=45)
    
    # 不同建筑结构价格
    plt.subplot(2, 2, 4)
    structure_price = df.groupby('建筑结构')['售价'].mean()
    sns.barplot(x=structure_price.index, y=structure_price.values)
    plt.title('不同建筑结构平均售价')
    plt.ylabel('平均售价(万元)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('floor_analysis.png')
    
    # 5. 聚类分析
    plt.figure(figsize=(12, 6))
    
    # 聚类分布
    plt.subplot(1, 2, 1)
    cluster_count = df['cluster'].value_counts()
    sns.barplot(x=cluster_count.index, y=cluster_count.values)
    plt.title('聚类分布')
    
    # 不同聚类价格
    plt.subplot(1, 2, 2)
    cluster_price = df.groupby('cluster')['售价'].mean()
    sns.barplot(x=cluster_price.index, y=cluster_price.values)
    plt.title('不同聚类平均售价')
    plt.ylabel('平均售价(万元)')
    
    plt.tight_layout()
    plt.savefig('cluster_analysis.png')
    
    # 6. 相关性分析
    plt.figure(figsize=(14, 12))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('数值特征相关性分析')
    plt.tight_layout()
    plt.savefig('correlation_analysis.png')
    
    return {
        'price_stats': df['售价'].describe().to_dict(),
        'unit_price_stats': df['单价'].describe().to_dict(),
        'area_stats': df['面积数值'].describe().to_dict(),
        'top_regions': region_price.head(5).to_dict(),
        'top_communities': df.groupby('小区')['售价'].mean().sort_values(ascending=False).head(10).to_dict(),
        'decoration_price': df.groupby('装修')['售价'].mean().to_dict(),
        'orientation_price': df.groupby('朝向')['售价'].mean().sort_values(ascending=False).head(5).to_dict(),
        'cluster_features': df.groupby('cluster')[['售价', '单价', '面积数值', '房间数']].mean().to_dict()
    }

# 执行分析
analysis_results = analyze_data(df)

# 保存分析结果
import json
with open('analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, ensure_ascii=False, indent=4)

print("\n分析完成，结果已保存到analysis_results.json")