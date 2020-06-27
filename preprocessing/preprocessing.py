from pydub import AudioSegment
import os
import numpy as np

def to_wav(file_path, songname=None, export_path=None):
    """songname이 None이면 file_path 폴더에 속한 전체 곡을 wav로 바꿔줌"""
    
    if(export_path==None):
            export_path=file_path

    if songname==None:
        for filename in os.listdir(file_path):
            
            if(filename[-3:]=="mp3"):
                sound=AudioSegment.from_mp3(file_path+"/"+filename)
                sound.export(export_path+"/"+filename[:-3]+"wav", format="wav")
                return export_path+"/"+filename[:-3]+"wav"
            elif (filename[-2:]=="au"):
                sound=AudioSegment.from_file(file_path+"/"+filename, "au")
                sound.export(export_path+"/"+filename[:-2]+"wav", format="wav")
                return export_path+"/"+filename[:-2]+"wav"
    else:
        if(songname[-3:]=="mp3"):
            sound=AudioSegment.from_mp3(file_path+"/"+songname)
            sound.export(export_path+"/"+songname[:-3]+"wav", format="wav")
            return export_path+"/"+songname[:-3]+"wav"
        elif (filename[-2:]=="au"):
            sound=AudioSegment.from_file(file_path+"/"+filename, "au")
            sound.export(export_path+"/"+songname[:-2]+"wav", format="wav")
            return export_path+"/"+songname[:-2]+"wav"


def normalize(vector):
    # Subtract the mean, and scale to the interval [-1,1]
    vector_minusmean = vector - vector.mean()
    return vector_minusmean/np.abs(vector_minusmean).max()