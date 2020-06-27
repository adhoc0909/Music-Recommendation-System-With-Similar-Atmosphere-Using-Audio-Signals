import librosa
from preprocessing.preprocessing import normalize
import csv
import os
import time
import numpy as np



def header_splitter(header, splitby):
    add=""
    res=""
    for i in range(splitby):
        add+="_"+str(i+1)+" "
    
    for tag in header:
        if(tag=="filename" or tag=="label"):
            res+=tag+" "
        else:
            for idx in add.split():
                res+=tag+idx+" "
    return res.split()



def timbral_feature_extraction(flname, split_by):

    header = "filename spectral_centroid_mean spectral_centroid_variance spectral_rolloff_mean spectral_rolloff_variance spectral_flux_mean spectral_flux_variance zero_crossing_rate_mean zero_crossing_rate_variance low_energy mfcc1_mean mfcc1_variance mfcc2_mean mfcc2_variance mfcc3_mean mfcc3_variance mfcc4_mean mfcc4_variance mfcc5_mean mfcc5_variance label"
    header=header_splitter(header.split(), split_by)

    file = open(flname, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

                        
    genres="blues classical country disco hiphop jazz metal pop reggae rock".split()

    cnt=10

    for g in genres:
        start=time.time()
        for filename in os.listdir(f'C:/Users/admin/OneDrive - 경희대학교/2020년 1학기 강의자료/데이터분석캡스톤디자인/프로젝트/데이터/gtzan-genre-collection/genres_wav/{g}'):
            songname = f'C:/Users/admin/OneDrive - 경희대학교/2020년 1학기 강의자료/데이터분석캡스톤디자인/프로젝트/데이터/gtzan-genre-collection/genres_wav/{g}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=30)
            y=normalize(y)
            
            
            to_append=[]
            centroid_mean=[]
            rolloff_mean=[]
            flux_mean=[]
            zcr_mean=[]
            mfcc1_mean=[]
            mfcc2_mean=[]
            mfcc3_mean=[]
            mfcc4_mean=[]
            mfcc5_mean=[]
            low_energy=[]
        
            centroid_variance=[]
            rolloff_variance=[]
            flux_variance=[]
            zcr_variance=[]
            mfcc1_variance=[]
            mfcc2_variance=[]
            mfcc3_variance=[]
            mfcc4_variance=[]
            mfcc5_variance=[]
           
        
            for i in range(split_by):
                centroid_mean.append(librosa.feature.spectral_centroid(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).mean())#spectral centroid
                rolloff_mean.append(librosa.feature.spectral_rolloff(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).mean()) #spectral rolloff
                flux_mean.append(librosa.onset.onset_strength(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).mean()) #spectral flux
                zcr_mean.append(librosa.feature.zero_crossing_rate(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)]).mean()) #zero crossing rate
                mfcc1_mean.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[0].mean()) #first five mfcc vectors
                mfcc2_mean.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[1].mean())
                mfcc3_mean.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[2].mean())
                mfcc4_mean.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[3].mean())
                mfcc5_mean.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[4].mean())
            
                centroid_variance.append(librosa.feature.spectral_centroid(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).var())#spectral centroid
                rolloff_variance.append(librosa.feature.spectral_rolloff(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).var()) #spectral rolloff
                flux_variance.append(librosa.onset.onset_strength(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).var()) #spectral flux
                zcr_variance.append(librosa.feature.zero_crossing_rate(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)]).var()) #zero crossing rate
                mfcc1_variance.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[0].var()) #first five mfcc vectors
                mfcc2_variance.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[1].var())
                mfcc3_variance.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[2].var())
                mfcc4_variance.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[3].var())
                mfcc5_variance.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[4].var())
            
                rmse=librosa.feature.rms(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], hop_length=sr)
                rmsmean=rmse.mean()
                low_energy.append(((y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)]<rmsmean).sum())/y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)].shape[0]) #low energy
        


            to_append.append(filename)
            to_append.extend(centroid_mean)
            to_append.extend(centroid_variance)
            to_append.extend(rolloff_mean)
            to_append.extend(rolloff_variance)
            to_append.extend(flux_mean)
            to_append.extend(flux_variance)
            to_append.extend(zcr_mean)
            to_append.extend(zcr_variance)
            to_append.extend(low_energy)
            to_append.extend(mfcc1_mean)
            to_append.extend(mfcc1_variance)
            to_append.extend(mfcc2_mean)
            to_append.extend(mfcc2_variance)
            to_append.extend(mfcc3_mean)
            to_append.extend(mfcc3_variance)
            to_append.extend(mfcc4_mean)
            to_append.extend(mfcc4_variance)
            to_append.extend(mfcc5_mean)
            to_append.extend(mfcc5_variance)
            to_append.append(g)


            
            
            file = open(flname, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append)
        print(g, "completed")
        t=(time.time()-start)
        cnt-=1
        print("time taken: %dm %ds"%(t//60, t%60))
        print("time expecting: %dm %ds"%((t*cnt)//60, (t*cnt)%60))
        print('\n')
        

def tempo_feature_extraction(flname, split_by):

    header=""
    for i in range(split_by):
        header+=str(i+1)+" "
    header=header.split()

    file = open(flname, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)


    genres="blues classical country disco hiphop jazz metal pop reggae rock".split()

    cnt=10

    for g in genres:
        start= time.time()
        for filename in os.listdir(f'C:/Users/admin/OneDrive - 경희대학교/2020년 1학기 강의자료/데이터분석캡스톤디자인/프로젝트/데이터/gtzan-genre-collection/genres_wav/{g}'):
            songname = f'C:/Users/admin/OneDrive - 경희대학교/2020년 1학기 강의자료/데이터분석캡스톤디자인/프로젝트/데이터/gtzan-genre-collection/genres_wav/{g}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=30)
            
            to_append=[]

            for i in range(split_by):
                to_append.append(float(librosa.beat.tempo(y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)))

            file = open(flname, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append)     

        print(g, "completed")
        t=(time.time()-start)
        cnt-=1
        print("time taken: %dm %ds"%(t//60, t%60))
        print("time expecting: %dm %ds"%((t*cnt)//60, (t*cnt)%60))
        print('\n')
        


def feature_extraction(songname, split_by, scaler, offset=0, duration=None):
    if(duration==None):
        y, sr = librosa.load(songname, mono=True, offset=offset)
    else:
        y, sr = librosa.load(songname, mono=True, offset=offset, duration=duration)
    y=normalize(y)
            
            
    to_append=[]
    centroid_mean=[]
    rolloff_mean=[]
    flux_mean=[]
    zcr_mean=[]
    mfcc1_mean=[]
    mfcc2_mean=[]
    mfcc3_mean=[]
    mfcc4_mean=[]
    mfcc5_mean=[]
    low_energy=[]
        
    centroid_variance=[]
    rolloff_variance=[]
    flux_variance=[]
    zcr_variance=[]
    mfcc1_variance=[]
    mfcc2_variance=[]
    mfcc3_variance=[]
    mfcc4_variance=[]
    mfcc5_variance=[]
           
        
    for i in range(split_by):
        centroid_mean.append(librosa.feature.spectral_centroid(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).mean())#spectral centroid
        rolloff_mean.append(librosa.feature.spectral_rolloff(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).mean()) #spectral rolloff
        flux_mean.append(librosa.onset.onset_strength(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).mean()) #spectral flux
        zcr_mean.append(librosa.feature.zero_crossing_rate(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)]).mean()) #zero crossing rate
        mfcc1_mean.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[0].mean()) #first five mfcc vectors
        mfcc2_mean.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[1].mean())
        mfcc3_mean.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[2].mean())
        mfcc4_mean.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[3].mean())
        mfcc5_mean.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[4].mean())
            
        centroid_variance.append(librosa.feature.spectral_centroid(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).var())#spectral centroid
        rolloff_variance.append(librosa.feature.spectral_rolloff(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).var()) #spectral rolloff
        flux_variance.append(librosa.onset.onset_strength(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr).var()) #spectral flux
        zcr_variance.append(librosa.feature.zero_crossing_rate(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)]).var()) #zero crossing rate
        mfcc1_variance.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[0].var()) #first five mfcc vectors
        mfcc2_variance.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[1].var())
        mfcc3_variance.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[2].var())
        mfcc4_variance.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[3].var())
        mfcc5_variance.append(librosa.feature.mfcc(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], sr=sr)[4].var())
            
        rmse=librosa.feature.rms(y=y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)], hop_length=sr)
        rmsmean=rmse.mean()
        low_energy.append(((y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)]<rmsmean).sum())/y[int(y.shape[0]/split_by)*i : int(y.shape[0]/split_by)*(i+1)].shape[0]) #low energy
    tempo=librosa.beat.tempo(y=y, sr=sr)


    to_append.extend(centroid_mean)
    to_append.extend(centroid_variance)
    to_append.extend(rolloff_mean)
    to_append.extend(rolloff_variance)
    to_append.extend(flux_mean)
    to_append.extend(flux_variance)
    to_append.extend(zcr_mean)
    to_append.extend(zcr_variance)
    to_append.extend(low_energy)
    to_append.extend(mfcc1_mean)
    to_append.extend(mfcc1_variance)
    to_append.extend(mfcc2_mean)
    to_append.extend(mfcc2_variance)
    to_append.extend(mfcc3_mean)
    to_append.extend(mfcc3_variance)
    to_append.extend(mfcc4_mean)
    to_append.extend(mfcc4_variance)
    to_append.extend(mfcc5_mean)
    to_append.extend(mfcc5_variance)
    to_append.append(tempo)
   

    to_append=np.array(to_append)

    
    

    X=scaler.transform(to_append.reshape(1,-1))

    return X