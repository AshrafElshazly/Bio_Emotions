import processing_Fextraction_ECG as PFecg
import processing_Fextraction_DREAMER as PFdreamer
import neurokit2 as nk
import pandas as pd
import scipy.io as sio
import model
from os.path import exists

heartRate    = 90
path_ecg     = "../Datasets/bio_resting_5min_100hz.csv"
path_dreamer = "../Datasets/DREAMER.mat"
path_test_emotion = "Data/HeartEmotions/sad.csv"
path_train = "Data/dataML_Modified.csv"

def PF_generated_ecg(heartRate):
    PFecg.plot_settings()
    ecg = nk.ecg_simulate(sampling_rate=256, heart_rate=heartRate)
    nk.signal_plot(ecg)
    signal ,info = PFecg.processing(ecg)
    nk.ecg_plot(signal[:3000], sampling_rate=256)
    data = nk.ecg_intervalrelated(signal)
    data = PFecg.customiz_data(data)
    data.to_csv("Data/HeartManual/generated_ECG_256hz.csv")
    return "Data/HeartManual/generated_ECG_256hz.csv"

def PF_ecg(path):
    PFecg.plot_settings()
    ecg = pd.read_csv(path)
    nk.signal_plot(ecg['ECG'])
    signal, info = nk.ecg_process(ecg["ECG"], sampling_rate=100)
    nk.ecg_plot(signal[:3000], sampling_rate=100)
    data = nk.ecg_intervalrelated(signal)
    data = PFecg.customiz_data(data)
    data.to_csv("Data/HeartManual/PF_ECG_100hz.csv")
    return "Data/HeartManual/PF_ECG_100hz.csv"

def PE_dreamer(path):
    raw = sio.loadmat(path)
    print("Dataset Load Done")
    print("  ")
    print("Processing and Extarctring Featuers....")
    df_ECG = PFdreamer.feat_extract_ECG(raw)
    print("Processing and Extarctring Featuers Done")
    df_features = pd.concat([df_ECG], axis=1)
    df_participant_affective = PFdreamer.participant_affective(raw)
    df_participant_affective["valence"] = (df_participant_affective
                                       ["valence"].astype(int))
    df_participant_affective["arousal"] = (df_participant_affective
                                           ["arousal"].astype(int))
    df_participant_affective["dominance"] = (df_participant_affective
                                             ["dominance"].astype(int))
    df = pd.concat([df_features, df_participant_affective], axis=1)
    
    data = df.loc[(df['target_emotion'] == 'calmness') |
                  (df['target_emotion'] == 'happiness') |
                  (df['target_emotion'] == 'sadness')|
                  (df['target_emotion'] == 'anger')|
                  (df['target_emotion'] == 'fear')|
                  (df['target_emotion'] == 'surprise')
                  ].copy()
    
    data['stress_bin'] = data['target_emotion'].map(
        {'calmness': 0, 'happiness': 1, 'sadness': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
        )
        
    data = PFdreamer.customiz_data(data)
    data.to_csv('Data/dataML_Modified.csv')
    
def recognition_emotion(train_data,path_test_emotion):
    X, y, groups, X_train, X_test, y_train, y_test = model.preparing_data(train_data)
    emotion = model.supervised_model(X, y, groups, X_test, y_test, path_test_emotion)
    print(emotion)
    
    
try:
    exists(path_train)
except:
    print("Trainning Data not found, Will generate new data")
    PE_dreamer(path_dreamer)
finally:
    x = input()
    if x == "test":
        path_ecg = path_test_emotion
    elif exists(path_ecg):
       path_ecg = PF_ecg(path_ecg)
    else:
        print("Generating data has been successfully")
        path_ecg = PF_generated_ecg(heartRate)
    recognition_emotion(path_train,path_ecg)