import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def plot_time():
    data = pd.read_csv('TaxiData-Sample.csv',header = None)
    data.columns = ['VehicleNum', 'Stime', 'Lng', 'Lat', 'OpenStatus', 'Speed']
    data.head(5)
    data['Hour'] = data['Stime'].apply(lambda r:r[:2])
    hourcount = data.groupby(data['Hour'])
    hourcount = hourcount['VehicleNum'].count().reset_index()
    sns.set_style('darkgrid',{'xtick.major.size':10,'ytick.major.size':10})

    fig     = plt.figure(1,(8,4),dpi = 300)    
    ax      = plt.subplot(111)
    plt.sca(ax)
    
    plt.bar(hourcount['Hour'],hourcount['VehicleNum'],width = 0.5)
    plt.title('Hourly data Volume')
    plt.ylim(0,80000)
    plt.ylabel('Data volume')
    plt.xlabel('Hour')
    plt.savefig(fname = 'test.svg',format = 'svg',bbox_inches = 'tight')
    plt.plot(hourcount['Hour'],hourcount['VehicleNum'],'k-',hourcount['Hour'],hourcount['VehicleNum'],'k.')
    st.markdown('\n每小时数据量')
    st.write(fig)
    dataod = pd.read_csv('TaxiOD.csv')
    dataod.columns = ['VehicleNum','Stime','SLng','SLat','ELng','ELat','Etime']
    dataod['Hour'] = dataod['Stime'].apply(lambda y:y.split(':')[0])
    hourcountod = dataod.groupby(dataod['Hour'])
    hourcountod = hourcountod['VehicleNum'].count().reset_index()
    sns.set_style('darkgrid',{'xtick.major.size':10,'ytick.major.size':10})

    fig     = plt.figure(1,(8,4),dpi = 300)    
    ax      = plt.subplot(111)
    plt.sca(ax)
    plt.plot(hourcountod['Hour'],hourcountod['VehicleNum'],'k-',hourcountod['Hour'],hourcountod['VehicleNum'],'k.')
    plt.bar(hourcountod['Hour'],hourcountod['VehicleNum'],width = 0.5)
    plt.title('Hourly order Volume')
    plt.ylim(0,30000)
    plt.ylabel('Order volume')
    plt.xlabel('Hour')
    plt.savefig(fname = 'od.svg',format = 'svg',bbox_inches = 'tight')
    st.markdown('\n每小时订单量')
    st.write(fig)

    dataod['order time'] = pd.to_datetime(dataod['Etime'])-pd.to_datetime(dataod['Stime'])
    dataod = dataod[-dataod['ELng'].isnull()]
    dataod['order seconds'] = dataod['order time'].apply(lambda r:r.seconds)

    fig     = plt.figure(1,(10,5),dpi = 250)    
    ax      = plt.subplot(111)
    plt.sca(ax)

    #只需要一行
    sns.boxplot(x="Hour", y=dataod["order seconds"]/60, data=dataod.sort_values(by=['Hour']),ax = ax)

    plt.ylabel('Order time(minutes)')
    plt.xlabel('Order start time')
    plt.ylim(0,60)
    st.markdown('\n每小时订单量-box图')
    st.write(fig)


@st.cache_data
def plot_space():
    import matplotlib as mpl
    import geopandas
    from shapely.geometry import Point,Polygon,shape
    import math
    shp = 'sz.shp'
    sz = geopandas.GeoDataFrame.from_file(shp,encoding = 'utf-8')

    szr = geopandas.GeoDataFrame.from_file('shenzhen_road.shp',encoding = 'utf-8')

    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    
    sz.plot(ax=ax)
    st.markdown('\n深圳行政区划')
    st.write(fig)
    plt.close()

    import math

    testlon = 114
    testlat = 22.5

    lon1 = 113.75194
    lon2 = 114.624187
    lat1 = 22.447837
    lat2 = 22.864748

    latStart = min(lat1, lat2)
    lonStart = min(lon1, lon2)

    accuracy = 500

    deltaLon = accuracy * 360 /(2 * math.pi * 6371004 * math.cos((lat1 + lat2) * math.pi / 360))
    deltaLat = accuracy * 360 /(2 * math.pi * 6371004)

    LONCOL = divmod(float(testlon) - (lonStart - deltaLon / 2), deltaLon)[0]
    LATCOL = divmod(float(testlat) - (latStart - deltaLat / 2), deltaLat)[0]

    HBLON = LONCOL * deltaLon + (lonStart - deltaLon / 2)
    HBLAT = LATCOL * deltaLat + (latStart - deltaLat /2)

    data = geopandas. GeoDataFrame()

    LONCOL1 = []
    LATCOL1 = []
    geometry = []
    HBLON1 = []
    HBLAT1 = []

    lonsnum = int((lon2-lon1)/deltaLon)+1
    latsnum = int((lat2-lat1)/deltaLat)+1

    for i in range(lonsnum):
        for j in range(latsnum):

            HBLON = i*deltaLon + (lonStart - deltaLon / 2)
            HBLAT = j*deltaLat + (latStart - deltaLat / 2)
            LONCOL1.append(i)
            LATCOL1.append(j)
            HBLON1.append(HBLON)
            HBLAT1.append(HBLAT)

            HBLON_1 = (i+1)*deltaLon + (lonStart - deltaLon / 2)
            HBLAT_1 = (j+1)*deltaLat + (latStart - deltaLat / 2)
            geometry.append(Polygon([\
            (HBLON-deltaLon/2,HBLAT-deltaLat/2),\
            (HBLON_1-deltaLon/2,HBLAT-deltaLat/2),\
            (HBLON_1-deltaLon/2,HBLAT_1-deltaLat/2),\
            (HBLON-deltaLon/2,HBLAT_1-deltaLat/2)]))


    data['LONCOL'] = LONCOL1
    data['LATCOL'] = LATCOL1
    data['HBLON'] = HBLON1
    data['HBLAT'] = HBLAT1
    data['geometry'] = geometry

    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    #data.plot(ax=ax)
    #st.write(fig)
    plt.close()
    grid = data[data.intersects(sz.unary_union)]
    grid.plot(ax=ax)
    st.markdown('\n栅格化')
    st.write(fig)
    plt.close()
    TaxiOD = pd.read_csv('TaxiOD.csv')
    TaxiOD.columns = ['VehicleNum','Stime','SLng','SLat','ELng','ELat','Etime']

    TaxiOD = TaxiOD[-TaxiOD['ELng'].isnull()].copy()

    TaxiOD['SLONCOL'] = ((TaxiOD['SLng'] - (lonStart - deltaLon / 2))/deltaLon).astype('int')
    TaxiOD['SLATCOL'] = ((TaxiOD['SLat'] - (latStart - deltaLat / 2))/deltaLat).astype('int')


    TaxiOD['SHBLON'] = TaxiOD['SLONCOL'] * deltaLon + (lonStart - deltaLon / 2)
    TaxiOD['SHBLAT'] = TaxiOD['SLATCOL'] * deltaLat + (latStart - deltaLat / 2)

    TaxiOD['ELONCOL'] = ((TaxiOD['ELng'] - (lonStart - deltaLon / 2))/deltaLon).astype('int')
    TaxiOD['ELATCOL'] = ((TaxiOD['ELat'] - (latStart - deltaLat / 2))/deltaLat).astype('int')

    TaxiOD['EHBLON'] = TaxiOD['ELONCOL'] * deltaLon + (lonStart - deltaLon / 2)
    TaxiOD['EHBLAT'] = TaxiOD['ELATCOL'] * deltaLat + (latStart - deltaLat / 2)

    TaxiOD = TaxiOD[(-((TaxiOD['SLONCOL'] == TaxiOD['ELONCOL']) & (TaxiOD['SLATCOL'] == TaxiOD['ELATCOL'])))]

    TaxiOD = TaxiOD[(TaxiOD['SLONCOL']>=0) & (TaxiOD['SLATCOL']>=0) &(TaxiOD['ELONCOL']>=0) & (TaxiOD['ELATCOL']>=0)&(TaxiOD['SLONCOL']<=lonsnum) & (TaxiOD['SLATCOL']<=latsnum) &(TaxiOD['ELONCOL']<=lonsnum) & (TaxiOD['ELATCOL']<=latsnum)]

    OD = TaxiOD.groupby(['SLONCOL','SLATCOL','ELONCOL','ELATCOL','SHBLON','SHBLAT','EHBLON','EHBLAT'])['VehicleNum'].count().rename('count').reset_index()

    fig = plt.figure(1,(10,8),dpi = 250)    
    ax = plt.subplot(111)
    plt.sca(ax)

    
    szr.plot(ax=ax,edgecolor = (0,0,0,0.5),linewidth = 0.3)
    SZ_all = geopandas.GeoDataFrame()
    SZ_all['geometry'] = [sz.unary_union]
    SZ_all.plot(ax = ax,edgecolor = (0,0,0,1),facecolor = (0,0,0,0),linewidths=0.5)

    OD1 = OD.sort_values(by = 'count')
    vmax = OD1['count'].max()
    norm = mpl.colors.Normalize(vmin=0,vmax=vmax)
    cmapname = 'spring_r'
    cmap = mpl.cm.get_cmap(cmapname)
  
    from shapely.geometry import LineString
    OD1g = geopandas.GeoDataFrame(OD1)
    OD1g['geometry'] = OD1g.apply(lambda r:LineString([[r['SHBLON'],r['SHBLAT']],[r['EHBLON'],r['EHBLAT']]]),axis = 1)
    OD1g.plot(ax = ax,column = 'count',vmax = vmax, vmin = 0,cmap = cmap,linewidth = (3 * OD1g['count']/ vmax))

    plt.axis('off')

    plt.imshow([[0,vmax+10]], cmap=cmap)
    #设定colorbar的大小和位置
    cax = plt.axes([0.08, 0.4, 0.02, 0.3])
    plt.colorbar(cax=cax)

    #然后要把镜头调整回到深圳地图那，不然镜头就在imshow那里了


    ax.set_xlim(113.6,114.8)
    ax.set_ylim(22.4,22.9)
    st.markdown('\n流向热度图')
    st.write(fig)
    plt.close()

    OD1g_f = pd.DataFrame(OD1g.groupby(['SLONCOL','SLATCOL'])['count'].sum().rename('freq')).reset_index()
    OD1g_f = OD1g_f.rename(columns = {'SLONCOL':'LONCOL','SLATCOL':'LATCOL'})

    freq_grid = pd.merge(grid,OD1g_f,on = ['LONCOL','LATCOL'])
    
    fig = plt.figure(1,(10,8),dpi = 250)    
    ax = plt.subplot(111)
    plt.sca(ax)

    grid.plot(ax =ax,edgecolor = (0,0,0,0.8),facecolor = (0,0,0,0),linewidths=0.2)
    SZ_all = geopandas.GeoDataFrame()
    SZ_all['geometry'] = [sz.unary_union]
    SZ_all.plot(ax = ax,edgecolor = (0,0,0,1),facecolor = (0,0,0,0),linewidths=0.5)

    fmax = freq_grid['freq'].max()
    norm_f = mpl.colors.Normalize(vmin=0,vmax=fmax)
    cmapname_f = 'autumn_r'
    cmap_f = mpl.cm.get_cmap(cmapname_f)

    freq_grid.plot(ax =ax,column = 'freq',cmap=cmap_f,linewidths=0.05)
    st.markdown('\n起点栅格热度图')
    st.write(fig)


    

@st.cache_data
def plot_train():
    import visuals as vs
    from sklearn import tree
    data = pd.read_csv('4train.csv')
    data.head()
    data.isnull().sum()
    data.describe()
    data['lginterval'] = np.log10(data['interval'])

    corr = data.corr()
    fig = plt.figure(figsize = (9,9))
    sns.heatmap(corr,annot=True,fmt='.2f',cmap='coolwarm')
    st.markdown('\n相关性热力图')
    st.write(fig)

    fig,ax = plt.subplots()
    sns.displot(data['interval'],kde=True)
    plt.title('Distrubution of interval')
    plt.xlabel('interval')
    plt.ylabel('Frequency')
    plt.show()
    st.markdown('\n订单时长分布图')
    st.write(fig)

    fig,ax = plt.subplots()
    sns.displot(data['lginterval'],kde=True)
    plt.title('Distrubution of lginterval')
    plt.xlabel('lginterval')
    plt.ylabel('Frequency')
    plt.show()
    st.markdown('\n订单时长对数分布图')
    st.write(fig)

    fig,ax = plt.subplots()
    sns.boxplot(y=data['lginterval'])
    plt.title('Boxplot of lginterval')
    plt.ylabel('lginterval')
    plt.show()
    st.markdown('\n订单时长对数box图')
    st.write(fig)

    fig = plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    sns.scatterplot(x=data['distance'],y=data['lginterval'])
    plt.title('lginterval vs distance')
    plt.subplot(1,2,2)
    sns.scatterplot(x=data['dtdis'],y=data['lginterval'])
    plt.title('lginterval vs dtdis')
    plt.tight_layout()
    plt.show()
    st.markdown('\n订单时长对数vs路径距离散点图')
    st.write(fig)
    
    fig,ax = plt.subplots()
    sns.barplot(data=data,x='isrushhour',y='interval',ax=ax)
    st.markdown('\n高峰期与否对时长影响')
    st.write(fig)
    
    
    from sklearn.model_selection import train_test_split

    prices=data['lginterval']
    features=data[['isrushhour','SLONCOL','SLATCOL','ELONCOL','ELATCOL','dtdis','distance']]
    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)

    from sklearn.metrics import r2_score
    dtr=tree.DecisionTreeRegressor()
    dtr.fit(X_train,y_train)
    regressor_predictions = dtr.predict(X_test)
    score=r2_score(y_test,regressor_predictions)
    print(f"r2_score for Decision Tree Classification: {score}")

    vs.ModelComplexity(X_train, y_train)

    from sklearn.model_selection import KFold,GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn import tree
    from sklearn import metrics

    def fit_model(X, y):
        """ 基于输入数据 [X,y]，利于网格搜索找到最优的决策树模型"""
        
        cross_validator = KFold()
        
        regressor = tree.DecisionTreeRegressor()

        params = {'max_depth':range(1,9)}
        
        scoring_fnc = make_scorer(metrics.r2_score)

        grid = GridSearchCV(regressor,params,scoring=scoring_fnc,cv=cross_validator) #,cross_validator
        
        # 基于输入数据 [X,y]，进行网格搜索
        grid = grid.fit(X, y)
        # 查看参数
        #print(pd.DataFrame(grid.cv_results_))
        # 返回网格搜索后的最优模型
        return grid.best_estimator_
    reg = fit_model(X_train,y_train)

    import joblib

    N_CORES = joblib.cpu_count(only_physical_cores=True)
    print(f"Number of physical cores: {N_CORES}")

    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, KFold

    models = {
        "Random Forest": RandomForestRegressor(
            min_samples_leaf=5, random_state=0, n_jobs=N_CORES
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_leaf_nodes=15, random_state=0, early_stopping=False
        ),
    }
    param_grids = {
        "Random Forest": {"n_estimators": [10, 20, 50, 100]},
        "Hist Gradient Boosting": {"max_iter": [10, 20, 50, 100, 300, 500]},
    }
    cv = KFold(n_splits=4, shuffle=True, random_state=0)

    results = []
    for name, model in models.items():
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            return_train_score=True,
            cv=cv,
        ).fit(X_train, y_train)
        result = {"model": name, "cv_results": pd.DataFrame(grid_search.cv_results_)}
        results.append(result)

        import plotly.colors as colors
    import plotly.express as px
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=["Train time vs score", "Predict time vs score"],
    )
    model_names = [result["model"] for result in results]
    colors_list = colors.qualitative.Plotly * (
        len(model_names) // len(colors.qualitative.Plotly) + 1
    )

    for idx, result in enumerate(results):
        cv_results = result["cv_results"].round(3)
        model_name = result["model"]
        param_name = list(param_grids[model_name].keys())[0]
        cv_results[param_name] = cv_results["param_" + param_name]
        cv_results["model"] = model_name

        scatter_fig = px.scatter(
            cv_results,
            x="mean_fit_time",
            y="mean_test_score",
            error_x="std_fit_time",
            error_y="std_test_score",
            hover_data=param_name,
            color="model",
        )
        line_fig = px.line(
            cv_results,
            x="mean_fit_time",
            y="mean_test_score",
        )

        scatter_trace = scatter_fig["data"][0]
        line_trace = line_fig["data"][0]
        scatter_trace.update(marker=dict(color=colors_list[idx]))
        line_trace.update(line=dict(color=colors_list[idx]))
        fig.add_trace(scatter_trace, row=1, col=1)
        fig.add_trace(line_trace, row=1, col=1)

        scatter_fig = px.scatter(
            cv_results,
            x="mean_score_time",
            y="mean_test_score",
            error_x="std_score_time",
            error_y="std_test_score",
            hover_data=param_name,
        )
        line_fig = px.line(
            cv_results,
            x="mean_score_time",
            y="mean_test_score",
        )

        scatter_trace = scatter_fig["data"][0]
        line_trace = line_fig["data"][0]
        scatter_trace.update(marker=dict(color=colors_list[idx]))
        line_trace.update(line=dict(color=colors_list[idx]))
        fig.add_trace(scatter_trace, row=1, col=2)
        fig.add_trace(line_trace, row=1, col=2)

    fig.update_layout(
        xaxis=dict(title="Train time (s) - lower is better"),
        yaxis=dict(title="Test R2 score - higher is better"),
        xaxis2=dict(title="Predict time (s) - lower is better"),
        legend=dict(x=0.72, y=0.05, traceorder="normal", borderwidth=1),
        title=dict(x=0.5, text="Speed-score trade-off of tree-based ensembles"),
    )
    st.write('模型对比评估')
    st.write(fig)

def demoselect():
    if set == '订单时间':
        plot_time()
    elif set == '订单空间':
        plot_space()
    else:
        plot_train()    

datasets = ('订单时间','订单空间','训练数据')

st.set_page_config(page_title="数据图示", page_icon="📈")


st.sidebar.header("数据图示")


st.markdown('# 数据图示')

set = st.sidebar.selectbox('选择数据集以查看图示',options=datasets,placeholder='选择你想查看有关图示的数据集')
demoselect()




st.write('')





# Streamlit 的部件会自动按顺序运行脚本。由于此按钮与任何其他逻辑都没有连接，因此它只会引起简单的重新运行。
st.button("重新运行")