import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import folium
from streamlit_folium import folium_static
import jieba
from wordcloud import WordCloud
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置页面配置
st.set_page_config(
    page_title="链家二手房数据分析",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载数据
@st.cache_data
def load_data():
    df = pd.read_csv('processed_houses.csv')
    return df

# 加载分析结果
@st.cache_data
def load_analysis_results():
    if os.path.exists('analysis_results.json'):
        with open('analysis_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# 主函数
def main():
    # 侧边栏
    st.sidebar.title("链家二手房数据分析")
    
    # 页面选择
    page = st.sidebar.selectbox(
        "选择页面",
        ["首页", "价格分析", "区域分析", "户型分析", "特征分析", "聚类分析", "预测模型", "数据探索"]
    )
    
    # 加载数据
    df = load_data()
    analysis_results = load_analysis_results()
    
    # 页面内容
    if page == "首页":
        show_homepage(df, analysis_results)
    elif page == "价格分析":
        show_price_analysis(df)
    elif page == "区域分析":
        show_region_analysis(df)
    elif page == "户型分析":
        show_house_type_analysis(df)
    elif page == "特征分析":
        show_feature_analysis(df)
    elif page == "聚类分析":
        show_cluster_analysis(df)
    elif page == "预测模型":
        show_prediction_model(df)
    elif page == "数据探索":
        show_data_exploration(df)

# 首页
def show_homepage(df, analysis_results):
    st.title("链家二手房数据分析")
    
    # 数据概览
    st.header("数据概览")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("数据量", f"{df.shape[0]}条")
    
    with col2:
        st.metric("平均售价", f"{df['售价'].mean():.2f}万元")
    
    with col3:
        st.metric("平均单价", f"{df['单价'].mean():.2f}元/㎡")
    
    with col4:
        st.metric("平均面积", f"{df['面积数值'].mean():.2f}㎡")
    
    # 价格分布
    st.subheader("价格分布")
    fig = px.histogram(df, x="售价", nbins=50, marginal="box", 
                      title="二手房售价分布",
                      labels={"售价": "售价(万元)"})
    st.plotly_chart(fig, use_container_width=True)
    
    # 区域分布
    st.subheader("区域分布")
    region_count = df['区域'].value_counts().reset_index()
    region_count.columns = ['区域', '数量']
    fig = px.bar(region_count.head(15), x='区域', y='数量', 
                title="各区域房源数量(Top 15)")
    st.plotly_chart(fig, use_container_width=True)
    
    # 户型分布
    st.subheader("户型分布")
    house_type_count = df['户型'].value_counts().reset_index()
    house_type_count.columns = ['户型', '数量']
    fig = px.pie(house_type_count.head(10), values='数量', names='户型', 
                title="户型分布(Top 10)")
    st.plotly_chart(fig, use_container_width=True)
    
    # 标题词云
    # 标题词云
    st.subheader("房源标题词云")
    
    @st.cache_data
    def generate_wordcloud(df):
        text = ' '.join(df['标题'].tolist())
        words = jieba.cut(text)
        words = ' '.join(words)
        
        wordcloud = WordCloud(
            font_path='simhei.ttf',  # 需要下载中文字体
            width=800,
            height=400,
            background_color='white'
        ).generate(words)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    
    try:
        fig = generate_wordcloud(df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"生成词云时出错: {e}")
        st.info("请确保已安装jieba、wordcloud库，并下载中文字体文件simhei.ttf")

# 价格分析页面
def show_price_analysis(df):
    st.title("价格分析")
    
    # 价格统计
    st.header("价格统计")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("售价统计(万元)")
        st.dataframe(df['售价'].describe())
    
    with col2:
        st.subheader("单价统计(元/㎡)")
        st.dataframe(df['单价'].describe())
    
    # 价格分布
    st.header("价格分布")
    
    tab1, tab2 = st.tabs(["售价", "单价"])
    
    with tab1:
        # 售价分布
        fig = px.histogram(df, x="售价", nbins=50, marginal="box", 
                          title="二手房售价分布",
                          labels={"售价": "售价(万元)"})
        st.plotly_chart(fig, use_container_width=True)
        
        # 售价箱线图
        fig = px.box(df, x="区域", y="售价", 
                    title="各区域售价分布",
                    labels={"售价": "售价(万元)", "区域": "区域"})
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # 单价分布
        fig = px.histogram(df, x="单价", nbins=50, marginal="box", 
                          title="二手房单价分布",
                          labels={"单价": "单价(元/㎡)"})
        st.plotly_chart(fig, use_container_width=True)
        
        # 单价箱线图
        fig = px.box(df, x="区域", y="单价", 
                    title="各区域单价分布",
                    labels={"单价": "单价(元/㎡)", "区域": "区域"})
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # 价格与面积关系
    st.header("价格与面积关系")
    
    fig = px.scatter(df, x="面积数值", y="售价", color="区域", 
                    title="面积与售价关系",
                    labels={"面积数值": "面积(㎡)", "售价": "售价(万元)"},
                    hover_data=["小区", "户型", "朝向"])
    st.plotly_chart(fig, use_container_width=True)
    
    # 价格与户型关系
    st.header("价格与户型关系")
    
    house_type_price = df.groupby('户型')['售价'].mean().reset_index().sort_values('售价', ascending=False)
    fig = px.bar(house_type_price.head(15), x='户型', y='售价', 
                title="各户型平均售价(Top 15)",
                labels={"售价": "平均售价(万元)", "户型": "户型"})
    st.plotly_chart(fig, use_container_width=True)
    
    # 价格与装修关系
    st.header("价格与装修关系")
    
    decoration_price = df.groupby('装修')['售价'].mean().reset_index()
    fig = px.bar(decoration_price, x='装修', y='售价', 
                title="各装修类型平均售价",
                labels={"售价": "平均售价(万元)", "装修": "装修类型"})
    st.plotly_chart(fig, use_container_width=True)

# 区域分析页面
def show_region_analysis(df):
    st.title("区域分析")
    
    # 区域房源数量
    st.header("区域房源数量")
    
    region_count = df['区域'].value_counts().reset_index()
    region_count.columns = ['区域', '数量']
    fig = px.bar(region_count, x='区域', y='数量', 
                title="各区域房源数量",
                labels={"数量": "房源数量", "区域": "区域"})
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 区域价格分析
    st.header("区域价格分析")
    
    tab1, tab2 = st.tabs(["平均售价", "平均单价"])
    
    with tab1:
        region_price = df.groupby('区域')['售价'].mean().reset_index().sort_values('售价', ascending=False)
        fig = px.bar(region_price, x='区域', y='售价', 
                    title="各区域平均售价",
                    labels={"售价": "平均售价(万元)", "区域": "区域"})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        region_unit_price = df.groupby('区域')['单价'].mean().reset_index().sort_values('单价', ascending=False)
        fig = px.bar(region_unit_price, x='区域', y='单价', 
                    title="各区域平均单价",
                    labels={"单价": "平均单价(元/㎡)", "区域": "区域"})
        st.plotly_chart(fig, use_container_width=True)
    
    # 小区分析
    st.header("小区分析")
    
    # 选择区域
    selected_region = st.selectbox("选择区域", df['区域'].unique())
    
    # 筛选数据
    filtered_df = df[df['区域'] == selected_region]
    
    # 小区房源数量
    community_count = filtered_df['小区'].value_counts().reset_index()
    community_count.columns = ['小区', '数量']
    fig = px.bar(community_count.head(15), x='小区', y='数量', 
                title=f"{selected_region}各小区房源数量(Top 15)",
                labels={"数量": "房源数量", "小区": "小区"})
    st.plotly_chart(fig, use_container_width=True)
    
    # 小区平均价格
    community_price = filtered_df.groupby('小区')['售价'].mean().reset_index().sort_values('售价', ascending=False)
    fig = px.bar(community_price.head(15), x='小区', y='售价', 
                title=f"{selected_region}各小区平均售价(Top 15)",
                labels={"售价": "平均售价(万元)", "小区": "小区"})
    st.plotly_chart(fig, use_container_width=True)

# 户型分析页面
def show_house_type_analysis(df):
    st.title("户型分析")
    
    # 户型分布
    st.header("户型分布")
    
    house_type_count = df['户型'].value_counts().reset_index()
    house_type_count.columns = ['户型', '数量']
    fig = px.pie(house_type_count.head(10), values='数量', names='户型', 
                title="户型分布(Top 10)")
    st.plotly_chart(fig, use_container_width=True)
    
    # 户型价格分析
    st.header("户型价格分析")
    
    house_type_price = df.groupby('户型')['售价'].mean().reset_index().sort_values('售价', ascending=False)
    fig = px.bar(house_type_price.head(15), x='户型', y='售价', 
                title="各户型平均售价(Top 15)",
                labels={"售价": "平均售价(万元)", "户型": "户型"})
    st.plotly_chart(fig, use_container_width=True)
    
    # 户型面积分析
    st.header("户型面积分析")
    
    house_type_area = df.groupby('户型')['面积数值'].mean().reset_index().sort_values('面积数值', ascending=False)
    fig = px.bar(house_type_area.head(15), x='户型', y='面积数值', 
                title="各户型平均面积(Top 15)",
                labels={"面积数值": "平均面积(㎡)", "户型": "户型"})
    st.plotly_chart(fig, use_container_width=True)
    
    # 朝向分析
    st.header("朝向分析")
    
    # 朝向分布
    orientation_count = df['朝向'].value_counts().reset_index()
    orientation_count.columns = ['朝向', '数量']
    fig = px.pie(orientation_count.head(10), values='数量', names='朝向', 
                title="朝向分布(Top 10)")
    st.plotly_chart(fig, use_container_width=True)
    
    # 朝向价格分析
    orientation_price = df.groupby('朝向')['售价'].mean().reset_index().sort_values('售价', ascending=False)
    fig = px.bar(orientation_price.head(10), x='朝向', y='售价', 
                title="各朝向平均售价(Top 10)",
                labels={"售价": "平均售价(万元)", "朝向": "朝向"})
    st.plotly_chart(fig, use_container_width=True)
    
    # 装修分析
    st.header("装修分析")
    
    # 装修分布
    decoration_count = df['装修'].value_counts().reset_index()
    decoration_count.columns = ['装修', '数量']
    fig = px.pie(decoration_count, values='数量', names='装修', 
                title="装修分布")
    st.plotly_chart(fig, use_container_width=True)
    
    # 装修价格分析
    decoration_price = df.groupby('装修')['售价'].mean().reset_index().sort_values('售价', ascending=False)
    fig = px.bar(decoration_price, x='装修', y='售价', 
                title="各装修类型平均售价",
                labels={"售价": "平均售价(万元)", "装修": "装修类型"})
    st.plotly_chart(fig, use_container_width=True)

# 特征分析页面
def show_feature_analysis(df):
    st.title("特征分析")
    
    # 相关性分析
    st.header("相关性分析")
    
    # 选择特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect(
        "选择要分析的特征",
        numeric_cols,
        default=['售价', '单价', '面积数值', '房间数', '厅数', '楼层数值']
    )
    
    if selected_features:
        # 计算相关性
        corr = df[selected_features].corr()
        
        # 绘制热力图
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                       title="特征相关性热力图")
        st.plotly_chart(fig, use_container_width=True)
    
    # 楼层分析
    st.header("楼层分析")
    
    # 楼层类型分布
    floor_type_count = df['楼层类型'].value_counts().reset_index()
    floor_type_count.columns = ['楼层类型', '数量']
    fig = px.pie(floor_type_count, values='数量', names='楼层类型', 
                title="楼层类型分布")
    st.plotly_chart(fig, use_container_width=True)
    
    # 楼层类型价格分析
    floor_type_price = df.groupby('楼层类型')['售价'].mean().reset_index()
    fig = px.bar(floor_type_price, x='楼层类型', y='售价', 
                title="各楼层类型平均售价",
                labels={"售价": "平均售价(万元)", "楼层类型": "楼层类型"})
    st.plotly_chart(fig, use_container_width=True)
    
    # 建筑结构分析
    st.header("建筑结构分析")
    
    # 建筑结构分布
    structure_count = df['建筑结构'].value_counts().reset_index()
    structure_count.columns = ['建筑结构', '数量']
    fig = px.pie(structure_count, values='数量', names='建筑结构', 
                title="建筑结构分布")
    st.plotly_chart(fig, use_container_width=True)
    
    # 建筑结构价格分析
    structure_price = df.groupby('建筑结构')['售价'].mean().reset_index().sort_values('售价', ascending=False)
    fig = px.bar(structure_price, x='建筑结构', y='售价', 
                title="各建筑结构平均售价",
                labels={"售价": "平均售价(万元)", "建筑结构": "建筑结构"})
    st.plotly_chart(fig, use_container_width=True)
    
    # 年份分析
    st.header("建筑年份分析")
    
    # 过滤掉年份为unknown的数据
    year_df = df[df['年份'] != 'unknown'].copy()
    # 提取年份数字
    year_df['年份数值'] = year_df['年份'].str.extract('(\d+)').astype(float)
    
    if not year_df.empty:
        # 年份分布
        fig = px.histogram(year_df, x="年份数值", nbins=20, 
                          title="建筑年份分布",
                          labels={"年份数值": "建筑年份"})
        st.plotly_chart(fig, use_container_width=True)
        
        # 年份与价格关系
        fig = px.scatter(year_df, x="年份数值", y="售价", 
                        title="建筑年份与售价关系",
                        labels={"年份数值": "建筑年份", "售价": "售价(万元)"},
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

# 聚类分析页面
def show_cluster_analysis(df):
    st.title("聚类分析")
    
    st.header("房源聚类分析")
    
    # 选择用于聚类的特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect(
        "选择用于聚类的特征",
        numeric_cols,
        default=['售价', '单价', '面积数值', '房间数', '厅数']
    )
    
    # 选择聚类数量
    n_clusters = st.slider("选择聚类数量", 2, 10, 5)
    
    if selected_features:
        # 准备数据
        X = df[selected_features].copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # 使用PCA降维以便可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # 创建可视化数据框
        pca_df = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'cluster': df['cluster']
        })
        
        # 绘制聚类结果
        fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='cluster',
                        title="房源聚类结果 (PCA降维后)",
                        labels={"PCA1": "主成分1", "PCA2": "主成分2", "cluster": "聚类"},
                        color_continuous_scale=px.colors.qualitative.G10)
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析各聚类的特征
        st.subheader("各聚类特征分析")
        
        cluster_analysis = df.groupby('cluster')[selected_features].mean().reset_index()
        st.dataframe(cluster_analysis)
        
        # 各聚类的房源数量
        cluster_count = df['cluster'].value_counts().reset_index()
        cluster_count.columns = ['聚类', '数量']
        fig = px.pie(cluster_count, values='数量', names='聚类', 
                    title="各聚类房源数量分布")
        st.plotly_chart(fig, use_container_width=True)
        
        # 各聚类的价格分布
        fig = px.box(df, x='cluster', y='售价', 
                    title="各聚类售价分布",
                    labels={"cluster": "聚类", "售价": "售价(万元)"})
        st.plotly_chart(fig, use_container_width=True)
        
        # 各聚类的区域分布
        cluster_region = pd.crosstab(df['cluster'], df['区域'])
        cluster_region_pct = cluster_region.div(cluster_region.sum(axis=1), axis=0) * 100
        
        fig = px.imshow(cluster_region_pct, 
                       title="各聚类区域分布(%)",
                       labels=dict(x="区域", y="聚类", color="百分比"))
        st.plotly_chart(fig, use_container_width=True)

# 预测模型页面
def show_prediction_model(df):
    st.title("房价预测模型")
    
    st.header("房价预测器")
    
    # 导入XGBoost
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    
    # 准备数据
    # 选择特征
    features = ['面积数值', '房间数', '厅数', '楼层数值', '是否南北通透', '装修']
    
    # 处理分类特征
    df_model = df.copy()
    
    # 处理缺失值
    for col in features:
        if col in df_model.columns and df_model[col].dtype in [np.float64, np.int64]:
            df_model[col] = df_model[col].fillna(df_model[col].mean())
    
    # 对装修进行编码
    if '装修' in features:
        le_decoration = LabelEncoder()
        df_model['装修_encoded'] = le_decoration.fit_transform(df_model['装修'].fillna('未知'))
        decoration_mapping = dict(zip(le_decoration.classes_, le_decoration.transform(le_decoration.classes_)))
        features[features.index('装修')] = '装修_encoded'
    
    # 准备训练数据
    X = df_model[features].copy()
    y = df_model['售价']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练XGBoost模型
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 显示模型性能
    st.subheader("模型性能")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("均方误差 (MSE)", f"{mse:.2f}")
    with col2:
        st.metric("决定系数 (R²)", f"{r2:.2f}")
    
    # 特征重要性
    st.subheader("影响房价的重要因素")
    
    # 获取特征重要性
    importance = model.feature_importances_
    feature_names = X.columns
    
    # 创建特征重要性数据框
    importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importance
    }).sort_values('重要性', ascending=False)
    
    # 绘制特征重要性条形图
    fig = px.bar(importance_df, x='特征', y='重要性', 
                title="影响房价的重要因素",
                labels={"重要性": "重要性得分", "特征": "房源特征"})
    
    # 替换特征名称为更友好的名称
    feature_friendly_names = {
        '面积数值': '面积',
        '房间数': '房间数量',
        '厅数': '客厅数量',
        '楼层数值': '楼层高度',
        '是否南北通透': '南北通透',
        '装修_encoded': '装修情况'
    }
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(feature_names))),
            ticktext=[feature_friendly_names.get(feat, feat) for feat in importance_df['特征']]
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 预测vs实际值
    st.subheader("预测准确性")
    pred_df = pd.DataFrame({
        '实际售价': y_test,
        '预测售价': y_pred
    })
    
    fig = px.scatter(pred_df, x='实际售价', y='预测售价', 
                    title="预测售价 vs 实际售价",
                    labels={"实际售价": "实际售价(万元)", "预测售价": "预测售价(万元)"},
                    trendline="ols")
    st.plotly_chart(fig, use_container_width=True)
    
    # 用户输入预测
    st.header("输入房源信息预测价格")
    st.write("请输入房源信息，我们将为您预测房价")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("房屋面积(平方米)", min_value=10.0, max_value=500.0, value=100.0, step=5.0)
        rooms = st.number_input("卧室数量", min_value=1, max_value=10, value=3, step=1)
        halls = st.number_input("客厅数量", min_value=0, max_value=5, value=2, step=1)
    
    with col2:
        floor = st.number_input("所在楼层", min_value=1, max_value=100, value=10, step=1)
        is_north_south = st.selectbox("是否南北通透", ["是", "否"])
        decoration = st.selectbox("装修情况", list(decoration_mapping.keys()))
    
    # 转换用户输入为模型输入
    is_north_south_value = 1 if is_north_south == "是" else 0
    decoration_value = decoration_mapping[decoration]
    
    # 创建输入数据框
    input_data = pd.DataFrame({
        '面积数值': [area],
        '房间数': [rooms],
        '厅数': [halls],
        '楼层数值': [floor],
        '是否南北通透': [is_north_south_value],
        '装修_encoded': [decoration_value]
    })
    
    # 预测按钮
    if st.button("预测房价"):
        # 预测
        prediction = model.predict(input_data)[0]
        
        # 显示预测结果
        st.success(f"预测售价: {prediction:.2f} 万元")
        
        # 提供价格区间
        lower_bound = prediction * 0.9
        upper_bound = prediction * 1.1
        st.info(f"考虑到市场波动，价格可能在 {lower_bound:.2f} 万元 到 {upper_bound:.2f} 万元 之间")
        
        # 提供一些建议
        st.subheader("购房建议")
        
        if area > 150:
            st.write("• 您选择的是大户型房源，适合大家庭居住，但维护成本较高")
        elif area < 60:
            st.write("• 您选择的是小户型房源，适合单身或小家庭，性价比较高")
        
        if is_north_south == "是":
            st.write("• 南北通透的房源通风采光较好，居住舒适度高")
        
        if decoration == "精装":
            st.write("• 精装修房源可以直接入住，但价格较高")
        elif decoration == "毛坯":
            st.write("• 毛坯房可以按照自己的喜好装修，但需要额外的装修费用和时间")

# 数据探索页面
def show_data_exploration(df):
    st.title("数据探索")
    
    # 数据概览
    st.header("数据概览")
    
    # 显示数据样本
    st.subheader("数据样本")
    st.dataframe(df.head())
    
    # 数据统计
    st.subheader("数据统计")
    st.dataframe(df.describe())
    
    # 数据类型
    st.subheader("数据类型")
    st.dataframe(pd.DataFrame(df.dtypes, columns=['数据类型']))
    
    # 缺失值分析
    st.subheader("缺失值分析")
    missing_data = pd.DataFrame({
        '缺失值数量': df.isnull().sum(),
        '缺失比例': df.isnull().sum() / len(df) * 100
    }).sort_values('缺失值数量', ascending=False)
    st.dataframe(missing_data)
    
    # 自定义查询
    st.header("自定义数据查询")
    
    # 选择筛选条件
    st.subheader("筛选条件")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 价格范围
        price_range = st.slider(
            "售价范围(万元)",
            float(df['售价'].min()),
            float(df['售价'].max()),
            (float(df['售价'].quantile(0.25)), float(df['售价'].quantile(0.75)))
        )
        
        # 面积范围
        area_range = st.slider(
            "面积范围(㎡)",
            float(df['面积数值'].min()),
            float(df['面积数值'].max()),
            (float(df['面积数值'].quantile(0.25)), float(df['面积数值'].quantile(0.75)))
        )
        
        # 房间数量
        room_options = sorted(df['房间数'].unique().tolist())
        selected_rooms = st.multiselect("房间数量", room_options, default=[2, 3])
        
        # 客厅数量
        hall_options = sorted(df['厅数'].unique().tolist())
        selected_halls = st.multiselect("客厅数量", hall_options, default=[1, 2])
    
    with col2:
        # 选择区域
        regions = ['不限'] + sorted(df['区域'].unique().tolist())
        selected_region = st.selectbox("选择区域", regions)
        
        # 选择朝向
        orientations = ['不限'] + sorted(df['朝向'].unique().tolist())
        selected_orientation = st.selectbox("选择朝向", orientations)
        
        # 选择装修
        decorations = ['不限'] + sorted(df['装修'].unique().tolist())
        selected_decoration = st.selectbox("选择装修", decorations)
        
        # 是否南北通透
        is_ns_options = ['不限', '是', '否']
        is_ns = st.selectbox("是否南北通透", is_ns_options)
    
    # 楼层偏好
    floor_types = ['不限'] + sorted(df['楼层类型'].unique().tolist())
    selected_floor_type = st.selectbox("楼层偏好", floor_types)
    
    # 重要性权重设置
    st.subheader("设置各因素的重要性")
    st.write("请拖动滑块设置各因素对您的重要程度（值越大表示越重要）")
    
    weight_price = st.slider("价格重要性", 0.0, 1.0, 0.8, 0.1)
    weight_area = st.slider("面积重要性", 0.0, 1.0, 0.7, 0.1)
    weight_rooms = st.slider("户型重要性", 0.0, 1.0, 0.6, 0.1)
    weight_location = st.slider("区域重要性", 0.0, 1.0, 0.5, 0.1)
    weight_decoration = st.slider("装修重要性", 0.0, 1.0, 0.4, 0.1)
    weight_orientation = st.slider("朝向重要性", 0.0, 1.0, 0.3, 0.1)
    
    # 推荐按钮
    if st.button("为我推荐房源"):
        # 开始筛选和计算相似度
        filtered_df = df.copy()
        
        # 基础筛选
        filtered_df = filtered_df[(filtered_df['售价'] >= price_range[0]) & (filtered_df['售价'] <= price_range[1])]
        filtered_df = filtered_df[(filtered_df['面积数值'] >= area_range[0]) & (filtered_df['面积数值'] <= area_range[1])]
        
        if selected_rooms:
            filtered_df = filtered_df[filtered_df['房间数'].isin(selected_rooms)]
        
        if selected_halls:
            filtered_df = filtered_df[filtered_df['厅数'].isin(selected_halls)]
        
        if selected_region != '不限':
            filtered_df = filtered_df[filtered_df['区域'] == selected_region]
        
        if selected_orientation != '不限':
            filtered_df = filtered_df[filtered_df['朝向'] == selected_orientation]
        
        if selected_decoration != '不限':
            filtered_df = filtered_df[filtered_df['装修'] == selected_decoration]
        
        if is_ns != '不限':
            is_ns_value = 1 if is_ns == '是' else 0
            filtered_df = filtered_df[filtered_df['是否南北通透'] == is_ns_value]
        
        if selected_floor_type != '不限':
            filtered_df = filtered_df[filtered_df['楼层类型'] == selected_floor_type]
        
        # 如果筛选后没有结果，放宽条件
        if len(filtered_df) == 0:
            st.warning("没有找到完全符合条件的房源，已为您放宽筛选条件")
            filtered_df = df[(df['售价'] >= price_range[0] * 0.8) & (df['售价'] <= price_range[1] * 1.2)]
            filtered_df = filtered_df[(filtered_df['面积数值'] >= area_range[0] * 0.8) & (filtered_df['面积数值'] <= area_range[1] * 1.2)]
        
        # 计算相似度得分
        if len(filtered_df) > 0:
            # 计算价格相似度（价格越接近中间值越好）
            price_mid = (price_range[0] + price_range[1]) / 2
            filtered_df['价格相似度'] = 1 - abs(filtered_df['售价'] - price_mid) / (price_range[1] - price_range[0] + 1)
            
            # 计算面积相似度
            area_mid = (area_range[0] + area_range[1]) / 2
            filtered_df['面积相似度'] = 1 - abs(filtered_df['面积数值'] - area_mid) / (area_range[1] - area_range[0] + 1)
            
            # 计算总相似度
            filtered_df['总相似度'] = (
                weight_price * filtered_df['价格相似度'] +
                weight_area * filtered_df['面积相似度']
            )
            
            # 如果有区域筛选，增加区域权重
            if selected_region != '不限':
                filtered_df['总相似度'] += weight_location
            
            # 如果有装修筛选，增加装修权重
            if selected_decoration != '不限':
                filtered_df['总相似度'] += weight_decoration
            
            # 如果有朝向筛选，增加朝向权重
            if selected_orientation != '不限':
                filtered_df['总相似度'] += weight_orientation
            
            # 按相似度排序
            filtered_df = filtered_df.sort_values('总相似度', ascending=False)
            
            # 显示推荐结果
            st.subheader(f"为您找到 {len(filtered_df)} 个匹配的房源")
            
            # 显示前10个推荐结果
            st.write("以下是最匹配您需求的房源:")
            
            # 创建一个更友好的显示数据框
            display_df = filtered_df[['标题', '售价', '单价', '小区', '区域', '户型', '面积', '朝向', '装修', '楼层', '总相似度']].head(10).copy()
            display_df['总相似度'] = display_df['总相似度'].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df)
            
            # 显示详细信息
            st.subheader("推荐房源详情")
            
            # 为前3个推荐结果创建选项卡
            top_houses = filtered_df.head(min(3, len(filtered_df)))
            tabs = st.tabs([f"推荐 {i+1}: {house['标题'][:15]}..." for i, house in enumerate(top_houses.to_dict('records'))])
            
            for i, tab in enumerate(tabs):
                house = top_houses.iloc[i]
                with tab:
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.subheader(house['标题'])
                        st.write(f"**价格:** {house['售价']}万元 ({house['单价']}元/㎡)")
                        st.write(f"**小区:** {house['小区']}")
                        st.write(f"**区域:** {house['区域']}")
                        st.write(f"**户型:** {house['户型']} | **面积:** {house['面积']}")
                        st.write(f"**朝向:** {house['朝向']} | **装修:** {house['装修']}")
                        st.write(f"**楼层:** {house['楼层']} | **建筑结构:** {house['建筑结构']}")
                        
                        if '年份' in house and house['年份'] != 'unknown':
                            st.write(f"**建筑年份:** {house['年份']}")
                        
                        st.write(f"**匹配度:** {house['总相似度']:.2%}")
                        
                        if '详情页' in house:
                            st.markdown(f"[查看详情]({house['详情页']})")
                    
                    with col2:
                        # 显示该房源与用户偏好的匹配度雷达图
                        categories = ['价格匹配度', '面积匹配度', '户型匹配度', 
                                     '区域匹配度', '装修匹配度', '朝向匹配度']
                        
                        # 计算各维度匹配度
                        price_match = house['价格相似度']
                        area_match = house['面积相似度']
                        room_match = 1.0 if selected_rooms and house['房间数'] in selected_rooms else 0.5
                        region_match = 1.0 if selected_region == '不限' or house['区域'] == selected_region else 0.5
                        decoration_match = 1.0 if selected_decoration == '不限' or house['装修'] == selected_decoration else 0.5
                        orientation_match = 1.0 if selected_orientation == '不限' or house['朝向'] == selected_orientation else 0.5
                        
                        values = [price_match, area_match, room_match, 
                                 region_match, decoration_match, orientation_match]
                        
                        # 创建雷达图
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='匹配度'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="房源匹配度分析"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"radar_chart_{i}")
            
            # 导出推荐结果
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="下载推荐结果",
                data=csv,
                file_name="recommended_houses.csv",
                mime="text/csv",
            )
        else:
            st.error("抱歉，没有找到符合条件的房源，请尝试放宽筛选条件")
# ... existing code ...

def show_recommendation_system(df):
    st.title("房源推荐系统")
    
    st.header("根据您的偏好推荐房源")
    st.write("请选择您理想房源的特征，我们将为您推荐最匹配的房源")
    
    # 创建多列布局用于用户输入
    col1, col2 = st.columns(2)
    
    with col1:
        # 价格范围
        price_range = st.slider(
            "售价范围(万元)",
            float(df['售价'].min()),
            float(df['售价'].max()),
            (float(df['售价'].quantile(0.25)), float(df['售价'].quantile(0.75)))
        )
        
        # 面积范围
        area_range = st.slider(
            "面积范围(㎡)",
            float(df['面积数值'].min()),
            float(df['面积数值'].max()),
            (float(df['面积数值'].quantile(0.25)), float(df['面积数值'].quantile(0.75)))
        )
        
        # 房间数量
        room_options = sorted(df['房间数'].unique().tolist())
        selected_rooms = st.multiselect("房间数量", room_options, default=[2, 3])
        
        # 客厅数量
        hall_options = sorted(df['厅数'].unique().tolist())
        selected_halls = st.multiselect("客厅数量", hall_options, default=[1, 2])
    
    with col2:
        # 选择区域
        regions = ['不限'] + sorted(df['区域'].unique().tolist())
        selected_region = st.selectbox("选择区域", regions)
        
        # 选择朝向
        orientations = ['不限'] + sorted(df['朝向'].unique().tolist())
        selected_orientation = st.selectbox("选择朝向", orientations)
        
        # 选择装修
        decorations = ['不限'] + sorted(df['装修'].unique().tolist())
        selected_decoration = st.selectbox("选择装修", decorations)
        
        # 是否南北通透
        is_ns_options = ['不限', '是', '否']
        is_ns = st.selectbox("是否南北通透", is_ns_options)
    
    # 楼层偏好
    floor_types = ['不限'] + sorted(df['楼层类型'].unique().tolist())
    selected_floor_type = st.selectbox("楼层偏好", floor_types)
    
    # 重要性权重设置
    st.subheader("设置各因素的重要性")
    st.write("请拖动滑块设置各因素对您的重要程度（值越大表示越重要）")
    
    weight_price = st.slider("价格重要性", 0.0, 1.0, 0.8, 0.1)
    weight_area = st.slider("面积重要性", 0.0, 1.0, 0.7, 0.1)
    weight_rooms = st.slider("户型重要性", 0.0, 1.0, 0.6, 0.1)
    weight_location = st.slider("区域重要性", 0.0, 1.0, 0.5, 0.1)
    weight_decoration = st.slider("装修重要性", 0.0, 1.0, 0.4, 0.1)
    weight_orientation = st.slider("朝向重要性", 0.0, 1.0, 0.3, 0.1)
    
    # 推荐按钮
    if st.button("为我推荐房源"):
        # 开始筛选和计算相似度
        filtered_df = df.copy()
        
        # 基础筛选
        filtered_df = filtered_df[(filtered_df['售价'] >= price_range[0]) & (filtered_df['售价'] <= price_range[1])]
        filtered_df = filtered_df[(filtered_df['面积数值'] >= area_range[0]) & (filtered_df['面积数值'] <= area_range[1])]
        
        if selected_rooms:
            filtered_df = filtered_df[filtered_df['房间数'].isin(selected_rooms)]
        
        if selected_halls:
            filtered_df = filtered_df[filtered_df['厅数'].isin(selected_halls)]
        
        if selected_region != '不限':
            filtered_df = filtered_df[filtered_df['区域'] == selected_region]
        
        if selected_orientation != '不限':
            filtered_df = filtered_df[filtered_df['朝向'] == selected_orientation]
        
        if selected_decoration != '不限':
            filtered_df = filtered_df[filtered_df['装修'] == selected_decoration]
        
        if is_ns != '不限':
            is_ns_value = 1 if is_ns == '是' else 0
            filtered_df = filtered_df[filtered_df['是否南北通透'] == is_ns_value]
        
        if selected_floor_type != '不限':
            filtered_df = filtered_df[filtered_df['楼层类型'] == selected_floor_type]
        
        # 如果筛选后没有结果，放宽条件
        if len(filtered_df) == 0:
            st.warning("没有找到完全符合条件的房源，已为您放宽筛选条件")
            filtered_df = df[(df['售价'] >= price_range[0] * 0.8) & (df['售价'] <= price_range[1] * 1.2)]
            filtered_df = filtered_df[(filtered_df['面积数值'] >= area_range[0] * 0.8) & (filtered_df['面积数值'] <= area_range[1] * 1.2)]
        
        # 计算相似度得分
        if len(filtered_df) > 0:
            # 计算价格相似度（价格越接近中间值越好）
            price_mid = (price_range[0] + price_range[1]) / 2
            filtered_df['价格相似度'] = 1 - abs(filtered_df['售价'] - price_mid) / (price_range[1] - price_range[0] + 1)
            
            # 计算面积相似度
            area_mid = (area_range[0] + area_range[1]) / 2
            filtered_df['面积相似度'] = 1 - abs(filtered_df['面积数值'] - area_mid) / (area_range[1] - area_range[0] + 1)
            
            # 计算总相似度
            filtered_df['总相似度'] = (
                weight_price * filtered_df['价格相似度'] +
                weight_area * filtered_df['面积相似度']
            )
            
            # 如果有区域筛选，增加区域权重
            if selected_region != '不限':
                filtered_df['总相似度'] += weight_location
            
            # 如果有装修筛选，增加装修权重
            if selected_decoration != '不限':
                filtered_df['总相似度'] += weight_decoration
            
            # 如果有朝向筛选，增加朝向权重
            if selected_orientation != '不限':
                filtered_df['总相似度'] += weight_orientation
            
            # 按相似度排序
            filtered_df = filtered_df.sort_values('总相似度', ascending=False)
            
            # 显示推荐结果
            st.subheader(f"为您找到 {len(filtered_df)} 个匹配的房源")
            
            # 显示前10个推荐结果
            st.write("以下是最匹配您需求的房源:")
            
            # 创建一个更友好的显示数据框
            display_df = filtered_df[['标题', '售价', '单价', '小区', '区域', '户型', '面积', '朝向', '装修', '楼层', '总相似度']].head(10).copy()
            display_df['总相似度'] = display_df['总相似度'].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df)
            
            # 显示详细信息
            st.subheader("推荐房源详情")
            
            # 为前3个推荐结果创建选项卡
            top_houses = filtered_df.head(min(3, len(filtered_df)))
            tabs = st.tabs([f"推荐 {i+1}: {house['标题'][:15]}..." for i, house in enumerate(top_houses.to_dict('records'))])
            
            for i, tab in enumerate(tabs):
                house = top_houses.iloc[i]
                with tab:
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.subheader(house['标题'])
                        st.write(f"**价格:** {house['售价']}万元 ({house['单价']}元/㎡)")
                        st.write(f"**小区:** {house['小区']}")
                        st.write(f"**区域:** {house['区域']}")
                        st.write(f"**户型:** {house['户型']} | **面积:** {house['面积']}")
                        st.write(f"**朝向:** {house['朝向']} | **装修:** {house['装修']}")
                        st.write(f"**楼层:** {house['楼层']} | **建筑结构:** {house['建筑结构']}")
                        
                        if '年份' in house and house['年份'] != 'unknown':
                            st.write(f"**建筑年份:** {house['年份']}")
                        
                        st.write(f"**匹配度:** {house['总相似度']:.2%}")
                        
                        if '详情页' in house:
                            st.markdown(f"[查看详情]({house['详情页']})")
                    
                    with col2:
                        # 显示该房源与用户偏好的匹配度雷达图
                        categories = ['价格匹配度', '面积匹配度', '户型匹配度', 
                                     '区域匹配度', '装修匹配度', '朝向匹配度']
                        
                        # 计算各维度匹配度
                        price_match = house['价格相似度']
                        area_match = house['面积相似度']
                        room_match = 1.0 if selected_rooms and house['房间数'] in selected_rooms else 0.5
                        region_match = 1.0 if selected_region == '不限' or house['区域'] == selected_region else 0.5
                        decoration_match = 1.0 if selected_decoration == '不限' or house['装修'] == selected_decoration else 0.5
                        orientation_match = 1.0 if selected_orientation == '不限' or house['朝向'] == selected_orientation else 0.5
                        
                        values = [price_match, area_match, room_match, 
                                 region_match, decoration_match, orientation_match]
                        
                        # 创建雷达图
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='匹配度'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="房源匹配度分析"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"data_exploration_{hash(str(fig.data))}")
            
            # 导出推荐结果
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="下载推荐结果",
                data=csv,
                file_name="recommended_houses.csv",
                mime="text/csv",
            )
        else:
            st.error("抱歉，没有找到符合条件的房源，请尝试放宽筛选条件")
# ... existing code ...

# 主函数
if __name__ == "__main__":
    main()