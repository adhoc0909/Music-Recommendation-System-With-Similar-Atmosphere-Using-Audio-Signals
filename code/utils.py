import pywt
import numpy as np
from sklearn.preprocessing import  StandardScaler, MinMaxScaler

import re

import os
import pandas as pd
import sys
import librosa
import time

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import librosa.display

from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

import os
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import requests
import pickle
from feature_extractor import feature_extraction
import warnings
warnings.filterwarnings('ignore')


def DTW(A, B, window=sys.maxsize, d=lambda x, y: abs(x - y)):
    # 비용 행렬 초기화
    A, B = np.array(A), np.array(B)
    M, N = len(A), len(B)
    cost = sys.maxsize * np.ones((M, N))

    # 첫번째 로우,컬럼 채우기
    cost[0, 0] = d(A[0], B[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(A[i], B[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(A[0], B[j])
    # 나머지 행렬 채우기
    for i in range(1, M):
        for j in range(max(1, i - window), min(N, i + window+1)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(A[i], B[j])

   
    return cost[-1, -1]

def vector_split(v, n_split):
    v=np.array(v)
    length=len(v)//n_split
    res=[]
    for i in range(n_split-1):
        res.append(v[i*length:(i+1)*length])
    res.append(v[(i+1)*length:])
    return np.array(res)

def split_mean(v, n_split):
    splitted_vector=vector_split(v, n_split)
    res=[]
    for i in splitted_vector:
        res.append(i.mean())
    return np.array(res)

def tone_comp(A, B):
    res=0
    for i in range(len(A)):
        res+=abs(A[i]-B[i])
    return res


def tonalityComparison(ch1, ch2):
    a=[]
    b=[]
    for i in range(len(ch1)):
        a.append(bigger_rate(ch1[i], 0.9))
        b.append(bigger_rate(ch2[i], 0.9))
    return tone_comp(a, b)

def tonalityComparision_with_rate(rate1, rate2):
    return tone_comp(rate1, rate2)

tonality=['aMajor', 'cMajor','cMinor','fMajor','gMinor']

def bigger_rate(data, threshold):
    data=np.array(data)
    cnt=0
    for i in data:
        if(i>=threshold):
            cnt+=1
    return cnt/data.shape[0]

def clean_str(text):
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    return text



def getFirstUrl(query):
    
    videoRoot="www.youtube.com/watch?v="
    URL = "https://www.youtube.com/results?search_query="+clean_str(query)
    
    a=-1
    
    while(a==-1):
        response = requests.get(URL) 
        response.status_code 
        a=response.text.find('data-context-item-id=')
    
    
    startIndex=a+len("data-context-item-id=")+1
    endIndex = response.text.find("\" ", startIndex)
    
    return videoRoot+response.text[startIndex:endIndex]

def recommend(quary_path, data_path):

   
    a=[]
  
    temp={}
    index=[]
    music=[]
    
    
    print("곡 로딩하겠습니다.")
    y, sr=librosa.load(quary_path, offset=30, duration=120, mono=True)
    #print("song loaded -- {:.2f}s".format(time.time()-start))
    print("곡 로딩하였습니다. 분석을 시작합니다. ")

    #start=time.time()
    mfcc_A=librosa.feature.mfcc(y, sr)
    A0=split_mean(mfcc_A[0], 80)
    A1=split_mean(mfcc_A[1], 80)
    A2=split_mean(mfcc_A[2], 80)
    A3=split_mean(mfcc_A[3], 80)
    A4=split_mean(librosa.feature.spectral_centroid(y)[0], 80)
    A5=split_mean(librosa.feature.spectral_rolloff(y)[0], 80)
    A6=split_mean(librosa.onset.onset_strength(y), 100)
    A7=split_mean(librosa.feature.zero_crossing_rate(y=y)[0], 80)
    A8=int(librosa.beat.tempo(y[:len(y)//6]))
    chroma_stft=librosa.feature.chroma_stft(y)
    rate=[]
    for i in range(12):
        rate.append(bigger_rate(chroma_stft[i], 0.9))
        
    A9=np.array(rate)
    
    A0_mean=A0.mean()
    A1_mean=A1.mean()
    A2_mean=A2.mean()
    A3_mean=A3.mean()
    A4_mean=A4.mean()
    A5_mean=A5.mean()
    A6_mean=A6.mean()
    A7_mean=A7.mean()
    
    print("곡 특징 파악 되었습니다. 데이터베이스 내의 곡들과 비교하여 유사한 곡 70곡을 골라냅니다.")
    
   
    for path in data_path:
        for file in os.listdir(path+'/feature1'):
            start=time.time()
            df=pd.read_csv(path+'/feature1'+'/'+file)


            B0_mean=(df['mfcc_0'].to_numpy()).mean()
            B1_mean=(df['mfcc_1'].to_numpy()).mean()
            B2_mean=(df['mfcc_2'].to_numpy()).mean()
            B3_mean=(df['mfcc_3'].to_numpy()).mean()
            B4_mean=(df['centroid'].to_numpy()).mean()
            B5_mean=(df['rolloff'].to_numpy()).mean()
            B6_mean=(df['flux'].to_numpy()).mean()
            B7_mean=(df['zcr'].to_numpy()).mean()
            B8=df['tempo'][0]
            B9=df['tonality_rate'].to_numpy()
            
            cost0=abs(A0_mean-B0_mean)
            cost1=abs(A1_mean-B1_mean)
            cost2=abs(A2_mean-B2_mean)
            cost3=abs(A3_mean-B3_mean)
            cost4=abs(A4_mean-B4_mean)
            cost5=abs(A5_mean-B5_mean)
            cost6=abs(A6_mean-B6_mean)
            cost7=abs(A7_mean-B7_mean)
            cost8=abs(A8-B8)
            cost9=tonalityComparision_with_rate(A9, B9)
            
            a.append([cost0, cost1, cost2, cost3, cost4, cost5, cost6, cost7, cost8, cost9])
            music.append(file[:-4])
            
            
    scaler=MinMaxScaler()
    scaler.fit(np.array(a))
    nor_a=scaler.transform(np.array(a))

    
    sum_a=[]
    for i in nor_a:
        sum_a.append((i[0]+i[1]+i[2]+i[3])/4+(i[4]+i[5]+i[6])/3+i[7]+i[8]+i[9])
    
    info={}

    for i in range(len(sum_a)):
        
        info[sum_a[i]]=music[i]
        

    rec=sorted(info)
    filtered_result=[]
    for filtered in rec[:70]:
        filtered_result.append(info[filtered])
    
    print("유사한 곡 70곡 검색하였습니다. 10곡 추천해드리겠습니다.")
####################################################################################################    
    
    
    
    
    cnt=0
    a=[]
    music=[]
    
    for file in filtered_result:
        file+='.csv'
        for path in data_path:
            if file not in os.listdir(path+'/feature1'):
                continue
                
            start=time.time()
            df=pd.read_csv(path+'/feature1'+'/'+file)


            B0=df['mfcc_0'].to_numpy()
            B1=df['mfcc_1'].to_numpy()
            B2=df['mfcc_2'].to_numpy()
            B3=df['mfcc_3'].to_numpy()
            B4=df['centroid'].to_numpy()
            B5=df['rolloff'].to_numpy()
            B6=df['flux'].to_numpy()
            B7=df['zcr'].to_numpy()
            B8=df['tempo'][0]
            B9=df['tonality_rate'].to_numpy()

            start=time.time()
            cost0 = fastdtw(A0, B0, dist=euclidean)[0]
            cost1 = fastdtw(A1, B1, dist=euclidean)[0]
            cost2 = fastdtw(A2, B2, dist=euclidean)[0]
            cost3 = fastdtw(A3, B3, dist=euclidean)[0]
            cost4 = fastdtw(A4, B4, dist=euclidean)[0]
            cost5 = fastdtw(A5, B5, dist=euclidean)[0]
            cost6 = fastdtw(A6, B6, dist=euclidean)[0]
            cost7 = fastdtw(A7, B7, dist=euclidean)[0]
            cost8 = abs(A8-B8)
            cost9 = tonalityComparision_with_rate(A9, B9)
            

            a.append([cost0, cost1, cost2, cost3, cost4, cost5, cost6, cost7, cost8, cost9])
            music.append(file[:-4])
            cnt+=1
            #print(cnt)
            #print(time.time()-start)
        

 

    scaler=MinMaxScaler()
    scaler.fit(np.array(a))
    nor_a=scaler.transform(np.array(a))

    
    sum_a=[]
    for i in nor_a:
        sum_a.append((i[0]+i[1]+i[2]+i[3])/4+(i[4]+i[5]+i[6])/3+i[7]+i[8]+i[9])
    
    info={}

    for i in range(len(sum_a)):
        
        info[sum_a[i]]=music[i]


    #print(info)
    #print(a)
    
    print('\n\n')
    
    rec=sorted(info)
    for i in rec[:10]:
    #for i in rec:
        print(info[i])


def genre_classifier(song):
    model = pickle.load(open('finalized_model_72.5.sav', 'rb')) #모델 load
    scaler=pickle.load(open('scaler.pkl', 'rb'))
        
    a=feature_extraction(song, 2, scaler, 30, 90)
    genre_dict={0:'blues', 1:'classical', 2:'country', 3:'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9:'rock'}

    return genre_dict[model.predict(a)[0]]
