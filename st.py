import streamlit as st
#import folium
#from streamlit_folium import st_folium
#from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import requests as req
import datetime
from datetime import time
import math
import joblib

with open('DTReg.pkl','rb') as file:
    D = joblib.load(file)

with open('HGBReg.pkl','rb') as file:
    H = joblib.load(file)


def getnal(kw,rsltnum):
    url='https://restapi.amap.com/v5/place/text?parameters'
    params = {'key':'1773ce2be463500a6a9086623099e421',
            'keywords':kw,
            'region':'深圳市',
            'city_limit':'true',
            }

    a = req.get(url=url,params=params)

    rtdict = eval(a.text)
    
    li = []
    for place in rtdict['pois'][:rsltnum]:
        li.append((place['name'],place['address'],place['location']))
        
        
    return pd.DataFrame(li,columns=['name','address','location'])



def getpath(lonlat1,lonlat2):
    # 服务地址
    host = "https://api.map.baidu.com"

    # 接口地址
    uri = "/directionlite/v1/driving"

    # 此处填写你在控制台-应用管理-创建应用后获取的AK
    ak = "MIYK6Pr6XijFB4LfaUxWxfxKqzSzb2DG"
    lonlat1 = lonlat1.split(',')[1]+','+lonlat1.split(',')[0]
    lonlat2 = lonlat2.split(',')[1]+','+lonlat2.split(',')[0]
    params = {
    "origin":    lonlat1,
    "destination":    lonlat2,
    "ak":       ak,

    }

    response = req.get(url = host + uri, params = params)
    
    if response:
        padict = eval(response.text)
        distance = padict['result']['routes'][0]['distance']
        duration = padict['result']['routes'][0]['duration']
    return distance,duration

def coord2grid(coord):
    testlon,testlat = coord.split(',')
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

    return LONCOL,LATCOL

def data_gather(sttime,SCOL,ECOL,coord1,coord2):
    isrushhour = 0
    StimeSec = sttime.hour * 3600 + sttime.minute * 60
    if (StimeSec>=25200 and StimeSec<32400) or (StimeSec>=64800 and StimeSec<72000):
        isrushhour = 1
    ELON,ELAT = ECOL
    SLON,SLAT = SCOL
    dtLON= np.abs(ELON-SLON)
    dtLAT= np.abs(ELAT-SLAT)
    dtdis = np.sqrt(np.square(dtLON)+np.square(dtLAT))
    distance,duration = getpath(coord1,coord2) 
    return StimeSec,isrushhour,SLON,SLAT,ELON,ELAT,dtdis,distance,duration

st.header('Dual-Model Time Prediction',)
st.markdown('---')

sz_center = (22.546053,114.025973)

#m = folium.Map(sz_center, zoom_start=11)
#fg = folium.FeatureGroup(name="marker")
           
# call to render Folium map in Streamlit
#st_data = st_folium(m,feature_group_to_add=fg,width=800,height=500) 

def click_button():
        st.session_state.clicked = True

st.markdown('set start time')
sttime = st.slider(label='start time',value=time(12,00),step=datetime.timedelta(minutes=1))
st.markdown('---')

st.markdown('search for POI(amap API) (!revoked, over limit!)')
rsltnum = st.slider('Result Number',1,10,5)
stp = st.text_input('Please enter start point')
dp = st.text_input('Please enter destination')

if 'clicked' not in st.session_state:
    st.session_state.clicked = False



st.button('Search', on_click=click_button,key=1)

if st.session_state.clicked:
    place1df = getnal(stp,rsltnum)
    st.write(place1df)
    option1 = st.selectbox('choose 1 start point',range(len(place1df)),key=3)
    
    place2df = getnal(dp,rsltnum)
    st.write(place2df)
    option2 = st.selectbox('choose 1 destination',range(len(place2df)),key=4)

st.markdown('---')
st.markdown('manual coordinate input')

coord1 = st.text_input('start point',placeholder='lat,lon')
coord2 = st.text_input('destination',placeholder='lat,lon')

if st.button('input'):
    test_data = data_gather(sttime=sttime,SCOL=coord2grid(coord1),ECOL=coord2grid(coord2),coord1=coord1,coord2=coord2)
    st.write(test_data)
    DTpredict = D.predict(np.array([test_data[:8]]))
    HGBpredict = H.predict(np.array([test_data[:8]]))
    st.markdown('Decision Tree Result')
    st.write(DTpredict)
    st.write(np.power(10,DTpredict))
    st.markdown('Hist Gradient Boosting Result')
    st.write(HGBpredict)
    st.write(np.power(10,HGBpredict))