# Project title (Capstone design 2020-2)

분위기가 유사한 음악 추천 시스템

2017110265 이호욱

## Overview

특정 음악을 듣다가, 듣고 있던 음악과 유사한 음악은 없는지 찾으려 할 때 찾는 방법에는 비슷한 아티스트의 음악을 찾거나 비슷한 장르의 음악을 찾는 방법이 있다. 하지만, 이는 아티스트나 장르에 대한 배경지식이 필요하여, 본인이 가지고 있는 배경지식 내에서만 유사한 음악을 찾을 수 있을 것이다. 따라서 이러한 배경지식 없이도 직관적으로 곡의 흐름, 멜로디, 분위기만으로 유사한 음악을 추천받을 수 있다면 좋을 것 같아 과제를 선정하게 되었다.

해당 프로젝트의 **프로세스**는 다음과 같다. 사용자가 프로그램에 유사한 음악을 찾고자 하는 음악을 쿼리를 하고, 추천 받고 싶은 장르를 선택한다. 이때 장르는 복수 선택이 가능하며, 선택하지 않는다면 본 프로젝트에서 구현된 학습된 장르 분류기 모델로 장르를 분류하여 해당 장르 내에서 유사도 비교 진행 후 유사한 곡을 추천해주는 방식이다.

장르 분류기 모델은 superviesd learning을 하였고, 학습을 위한 특징으로는 음악의 음색(주파수영역), 템포, 그 외에 time-domain에서 추출할 수 있는 시끄러운 정도 등을 사용하였다. 다만, 음악의 장르는 fuzzy한 특성을 가지고 있기 때문에, 선행연구에서도 분류 정확도가 70% 이내로 나오는 것을 확인하였다. 장르 분류가 정확하지 않다면 추천 시 다른 장르에서 추천하게 되어 좋은 추천이 되지 않을 수도 있다는 점을 반영하여, 사용자가 원한다면 해당모델을 사용하고, 원하지 않는다면 사용하지 않도록 설계하였다.

사용한 특징
1. Spectral Centroid
2. Spectral rolloff
4. Spectral flux
5. Low Energy
6. MFCC
7. Zero Crossing rate
8. Tempo
9. Chroma level


프로그램 성능을 높이기 위해 한 일

1. 벡터 압축 : 유사도 비교를 위해 사용한 특징은 장르분류모델에서 사용한 음색, 템포, time-domain에서 추출한 특징 외에도 음계의 진행 및 조성(tonality)을 사용하였다. 다만 일반적으로 44.1KHz 또는 22.05KHz의 sampling rate을 가지고 있다는 음악의 특성 상 stft연산(20ms의 윈도우 프레임)을 진행하면 천 단위의 차원을 가진 벡터가 추출된다. 또한 시계열 데이터 특성 상 단순한 euclidean distance를 유사도 척도로 두게 된다면 문제가 생기므로, 시간복잡도가 O(n^2)인 DTW(Dynamic Time Warping) 알고리즘을 유사도 척도로 두게 되었다. 이럴 경우 생기는 문제는 데이터베이스에 가지고 있는 음악의 수가 많아질수록 검색시간이 상당히 오래걸린다는 문제가 생긴다.
이를 해결하기 위해 각 특징벡터들을 80개의 벡터로 슬라이싱한 후 각각의 평균값을 구해 80차원의 벡터로 압축한 후 DTW연산을 하여 검색시간 단축에 기여한다. 

2.  빠른 검색으로 대략 유사한 곡 필터링 : DTW 연산을 하기 앞서 각 시계열 데이터인 특징벡터의 평균값을 우선 euclidian distance로 유사도 비교를 진행하여 후보곡 약 70곡정도를 먼저 뽑아놓고, 해당 곡들중에서
DTW연산을 하여 유사도를 비교하여 검색시간 단축에 기여한다. 

4. 음계 진행 및 조성(tonality)을 비교하기 위한 알고리즘 구현. 곡의 조성은 곡의 분위기(minor: 어두운, major: 밝은 등)를 결정하는 중요한 요소이므로, 이를 위해 각 chroma level(도, 도#, 레, 레#, 미, 파, 파#, 솔, 솔#, 라, 라#, 시)의 비율들을 계산하여 이를 비교한다. 

## Schedule
| Contents             | March | April |  May  | June  |
|----------------------|-------|-------|-------|-------|
|  서베이 및 논문 열람  |   o   |   o   |   o   |       |
|  데이터셋 확보        |       |   o   |       |       |
|  데이터 전처리        |       |   o   |   o   |       |
|  프로그램 전반 설계   |       |   o   |       |       |
|  장르 예측 모델 구현  |       |   o   |   o   |   o   |
|  추천 모델 구현       |       |   o   |   o   |   o   |
|  프로그램 성능 테스트 |       |    o  |   o   |   o   |


## UI
본 프로젝트의 프로그램은 사용자의 직관성과 편의성을 위해 GUI 방식으로 구현하였다. 이를 위해 파이썬의 PyQt5를 통해 구현하였다.

![3](https://user-images.githubusercontent.com/52408669/85917524-a7054600-b895-11ea-86f5-32250e3b1d55.PNG)





## Results
#### 장르분류기
장르 분류를 위해 사용한 특징은 음악에서 추출한 Spectral centroid, Spectral rolloff, Spectral flux, Low energy, MFCC, Zero crossing rate, Tempo로, tempo를 제외한 나머지 특징벡터를 여러개의 벡터로 쪼갠 후, 쪼개진 벡터들의 mean값과 variance 값을 특징벡터로 가졌다. 이 때 몇 개로 쪼갤 것인지 결정하기 위해, 모델을 학습하여 언제 가장 testa accuracy가 가장 높은지를 측정해았고, 그 결과 2개로 쪼갰을 때의 test accuracy가 가장 높게 나왔다. 

이에 따라 최대 약 65% ~ 70%의 정확도가 도출되었다.




#### 프로그램 전반에 대한 MOS 평가
프로그램 평가를 위해 MOS 평가를 진행했다. MOS 평가 질문으로는 다음과 같다. 

1. 매우 유사
2. 약간 유사
3. 별로 유사하지 않음
4. 아예 다름

총 104개의 응답을 얻었고, 평균 2.28점으로 "약간 유사"에 가까운 점수를 얻었다.  


## Conclusion

#### 한계점
1. 너무 긴 검색시간을 갖는다. 한 번 추천을 받기 위해서는 약 1분의 시간이 소요되는데, 이는 사용자가 느끼기에 불편할 것이라 생각된다. 
2. 데이터베이스가 빈약하다. 실제 스트리밍을 서비스하는 회사의 경우 자사에 큰 음원 데이터베이스를 구축하여 서비스를 하지만, 이 프로젝트를 진행하면서 이는 현실적으로 불가능하기 때문에, 약 3천개 가량의 음악을 youtube-dl 프로그램을 통해 직접 유튜브에 있는 음악들을 다운로드 받았다. 만일 추천받고 싶은 음악과 비슷한 음악이 데이터베이스에 없다면 추천 성능이 뛰어날지라도 비슷한 음악을 추천할 수 없다는 점에서 이는 한계점이라 할 수 있다. 
3. 검증되지 않은 조성 비교 알고리즘을 사용하였다. 조성을 정교하게 비교하고자 한다면, 곡의 조성 레이블을 분류하는 모델을 학습시켜, 분류된 조성간의 거리 비교를 해야한다. 하지만 본 프로젝트에서는 오로지 각 크로마의 비율만을 비교하였기 때문에, 조성간 거리를 정확하게 비교하리라는 보장은 없다.

#### 기대효과
 위에서 언급된 한계점을 해결하게 된다면, 기존의 협업기반 방식에 더불어 내용기반 추천시스템을 통해 정교한 추천을 할 수 있을 것이다.  그 결과 사용자의 만족도를 높일 수 있고, 이는 스트리밍 산업에 기여를 할 수 있을 것이라 생각된다. 
 
 ## Discussion
  본 프로젝트를 끝내고 아쉬운 부분에 대해 다음 연구를 제안한다.
  
 1. 시간 복잡도가 낮은 시계열 데이터 유사도 검색 방식으로 DTW 알고리즘을 대체
 2. 클러스터링 등을 하여 전체 DB 중 검색할 데이터 자체를 줄이는 방식
 3. 특징 벡터의 크기 및 차원을 축소하는 방식(PCA 등)
 4. 조성(tonality) 분류기를 학습하여, 두 음악 간의 조성의 거리를 비교하는 방식
 
 ## Report
 최종보고서 https://drive.google.com/file/d/1Pzkwq5bgVIRRZ_yipwlL0kwq_V2BcuEf/view?usp=sharing
 
 demo https://drive.google.com/file/d/14FOvfkVt-oT_lgdW7OyFLHGI6mFBGG3f/view?usp=sharing


 ## Example Code
 

def recommend(quary_path, data_path):

   
    a=[]
  
    temp={}
    index=[]
    music=[]
    
    
    y, sr=librosa.load(quary_path, offset=30, duration=120, mono=True) #곡 로딩
    
    ## input 곡 특징 추출
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

    
   
    for path in data_path:
        for file in os.listdir(path+'/feature1'):
         
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
    
    # 유사곡 70곡 필터링

    
    # 필터링 된 70곡 DTW 
    
    
    cnt=0
    a=[]
    music=[]
    
    for file in filtered_result:
        file+='.csv'
        for path in data_path:
            if file not in os.listdir(path+'/feature1'):
                continue
                
   
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
    for i in rec[:10]:
        print(info[i]) #10곡 추천

