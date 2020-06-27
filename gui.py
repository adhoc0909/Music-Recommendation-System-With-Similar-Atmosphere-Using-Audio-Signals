
import numpy as np
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from utils import *

from PyQt5 import QtCore, QtGui, QtWidgets 
import os

import pandas as pd
import sys
import librosa
import time

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw


#class ThreadClass(QtCore.QThread): 
#    def __init__(self, parent = None): 
#        super(ThreadClass,self).__init__(parent)
#    def run(self): 
        


class Ui_Form(object):

    

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(846, 684)

        self.path = "C:/Users/admin/OneDrive - 경희대학교/2020년 1학기 강의자료/데이터분석캡스톤디자인/프로젝트/데이터/음악"
        self.data_path = "C:/Users/admin/OneDrive - 경희대학교/2020년 1학기 강의자료/데이터분석캡스톤디자인/프로젝트/데이터/유튜브"
        self.filelist = os.listdir(self.path)
        self.genrepath = "C:/Users/admin/OneDrive - 경희대학교/2020년 1학기 강의자료/데이터분석캡스톤디자인/프로젝트/데이터/유튜브"
        self.genrelist = os.listdir(self.genrepath)

        self.scrollArea = QtWidgets.QScrollArea(Form)
        self.scrollArea.setGeometry(QtCore.QRect(10, 30, 351, 311))
        self.scrollArea.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.scrollArea.setAutoFillBackground(False)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 349, 309))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        
        self.radioButton=[]
        for i in range(len(self.filelist)):
            self.radioButton.append(QtWidgets.QRadioButton(self.scrollAreaWidgetContents))
            self.radioButton[i].setObjectName("radioButton")
            self.verticalLayout_2.addWidget(self.radioButton[i])

        
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea_2 = QtWidgets.QScrollArea(Form)
        self.scrollArea_2.setGeometry(QtCore.QRect(10, 360, 351, 261))
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 349, 259))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.gridLayout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout.setObjectName("gridLayout")

        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(20, 10, 250, 15))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(20, 344, 250, 15))
        self.label_2.setObjectName("label_2")

        self.checkBox=[]
        for i in range(len(self.genrelist)):
            self.checkBox.append(QtWidgets.QCheckBox(self.scrollAreaWidgetContents_2))
            self.checkBox[i].setObjectName("checkBox")
            self.gridLayout.addWidget(self.checkBox[i], i, 0, 1, 1)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.listView = QtWidgets.QListView(Form)
        self.listView.setGeometry(QtCore.QRect(370, 30, 411, 601))
        self.listView.setObjectName("listView")
        
        
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(140, 640, 93, 28))
        self.pushButton.setObjectName("pushButton")

        


   
        self.toolButton = QtWidgets.QToolButton(Form)
        self.toolButton.setGeometry(QtCore.QRect(700, 640, 81, 31))
        self.toolButton.setObjectName("toolButton")


      
        self.listWidget = QtWidgets.QListWidget(Form)
        self.listWidget.setGeometry(QtCore.QRect(410, 120, 341, 451))
        self.listWidget.setObjectName("listWidget")
        

        
    

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        self.pushButton.clicked.connect(self.retrieve_items)


        

    def retrieve_items(self):
        
        self.input_song = "empty"
        self.genre_selected=[]
        for i in range(len(self.filelist)):
            if self.radioButton[i].isChecked():
                self.input_song = self.path+"/"+self.filelist[i]
                break
        for i in range(len(self.genrelist)):
            if self.checkBox[i].isChecked():
                self.genre_selected.append(self.data_path+"/"+self.genrelist[i])



        if(self.input_song!="empty" and self.input_song[-4:]==".mp3" and self.genre_selected!=[]):
            self.recommend(self.input_song, self.genre_selected)
        elif self.genre_selected==[]:
            self.recommend(self.input_song, [self.data_path+"/"+genre_classifier(self.input_song)])
    

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))

        for i in range(len(self.filelist)):
            self.radioButton[i].setText(_translate("Form", self.filelist[i][:-4]))
        
        for i in range(len(self.genrelist)):
            self.checkBox[i].setText(_translate("Form", self.genrelist[i]))
        self.pushButton.setText(_translate("Form", "분석"))

        self.label.setText(_translate("Form", "추천 받을 음악을 선택해주세요"))
        self.label_2.setText(_translate("Form", "추천 받을 장르를 선택해주세요"))
        self.toolButton.setText(_translate("Form", "reset"))

       
    def recommend(self, quary_path, data_path):

    
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


    
        print('\n\n')
       
        rec=sorted(info)
        idx=0

        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)

        for i in rec[:10]:
            item = QtWidgets.QListWidgetItem()
            self.listWidget.addItem(item)
            itemq = self.listWidget.item(idx)
            itemq.setText('->'+info[i])

            idx+=1

            item = QtWidgets.QListWidgetItem()
            self.listWidget.addItem(item)
            itemq = self.listWidget.item(idx)
            itemq.setText(getFirstUrl(clean_str(info[i])))
            
            idx+=1

            item = QtWidgets.QListWidgetItem()
            self.listWidget.addItem(item)
            itemq = self.listWidget.item(idx)
            itemq.setText('---------------------')

            idx+=1


        
        self.listWidget.setSortingEnabled(__sortingEnabled)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    current_exit_code = 0

    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
