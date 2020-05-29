# Project title (Capstone design 2020-2)

분위기가 유사한 음악 추천 시스템

2017110265 이호욱

## Overview

특정 음악을 듣다가, 듣고 있던 음악과 유사한 음악은 없는지 찾으려 할 때 찾는 방법에는 비슷한 아티스트의 음악을 찾거나 비슷한 장르의 음악을 찾는 방법이 있다. 하지만, 이는 아티스트나 장르에 대한 배경지식이 필요하여, 본인이 가지고 있는 배경지식 내에서만 유사한 음악을 찾을 수 있을 것이다. 따라서 이러한 배경지식 없이도 직관적으로 곡의 흐름, 멜로디, 분위기만으로 유사한 음악을 추천받을 수 있다면 좋을 것 같아 과제를 선정하게 되었다.

해당 프로젝트의 **프로세스**는 다음과 같다. 사용자가 프로그램에 유사한 음악을 찾고자 하는 음악을 쿼리를 하고, 추천 받고 싶은 장르를 선택한다. 이때 장르는 복수 선택이 가능하며, 선택하지 않는다면 본 프로젝트에서 구현된 학습된 장르 분류기 모델로 장르를 분류하여 해당 장르 내에서 유사도 비교 진행 후 유사한 곡을 추천해주는 방식이다.

장르 분류기 모델은 superviesd learning을 하였고, 학습을 위한 특징으로는 음악의 음색(주파수영역), 템포, 그 외에 time-domain에서 추출할 수 있는 시끄러운 정도 등을 사용하였다. 다만, 음악의 장르는 fuzzy한 특성을 가지고 있기 때문에, 선행연구에서도 분류 정확도가 70% 이내로 나오는 것을 확인하였다. 이 특성을 반영하여, 사용자가 원한다면 해당모델을 사용하고, 원하지 않는다면 사용하지 않도록 설계할 계획이다.


프로그램 성능을 높이기 위해 할 일

1. 벡터 압축 : 유사도 비교를 위해 사용한 특징은 장르분류모델에서 사용한 음색, 템포, time-domain에서 추출한 특징 외에도 음계의 진행 및 조성(tonality)을 사용하였다. 다만 일반적으로 44.1KHz 또는 22.05KHz의 sampling rate을 가지고 있다는 음악의 특성 상 stft연산(20ms의 윈도우 프레임)을 진행하면 천 단위의 차원을 가진 벡터가 추출된다. 또한 시계열 데이터 특성 상 단순한 euclidean distance를 유사도 척도로 두게 된다면 문제가 생기므로, 시간복잡도가 O(n^2)인 DTW(Dynamic Time Warping) 알고리즘을 유사도 척도로 두게 되었다. 이럴 경우 생기는 문제는 데이터베이스에 가지고 있는 음악의 수가 많아질수록 검색시간이 상당히 오래걸린다는 문제가 생긴다.
이를 해결하기 위해 각 특징벡터들을 80개의 벡터로 슬라이싱한 후 각각의 평균값을 구해 80차원의 벡터로 압축한 후 DTW연산을 하여 검색시간 단축에 기여한다. 

2. 군집화 : 음악의 특징에 따라 군집화를 하여 우선 음악들을 비지도 분류를 한다면, 검색해야 할 곡들이 적어지기 때문에 검색시간 단축에 기여한다. 

3. 필터 : DTW 연산을 하기 앞서 각 시계열 데이터의 평균값을 우선 euclidian distance로 유사도 비교를 진행하여 후보곡 약 70곡정도를 먼저 뽑아놓고, 해당 곡들중에서
DTW연산을 하여 유사도를 비교하여 검색시간 단축에 기여한다. 

4. 음계 진행 및 조성(tonality)을 비교하기 위한 알고리즘 구현. 곡의 조성은 곡의 분위기(minor: 어두운, major: 밝은 등)를 결정하는 중요한 요소이므로, 이를 위해 정교한 알고리즘을 구현한다. 

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




## Results


## Conclusion


## Reports
