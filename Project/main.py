import model
from os.path import exists
import joblib
import dreamer
import ecg as PFecg
import numpy as np
import pandas as pd


def run(train_path: str, dreamer_path: str):

    if(not exists(train_path)):
        print("Trainning Data not found, Will generate new data")
        if(exists(dreamer_path)):
            dreamer.PE_dreamer(dreamer_path)
        else:
            return "DREAMER Dataset Not Found"

    model_saved, result = model.supervised_model(train_path)
    joblib.dump(model_saved, 'model.pkl')
    return result


def app(**kwargs):
    test_path = kwargs.get('test', "")
    ecg_path = kwargs.get('ecg', "")
    heartrate = kwargs.get('heartrate', 80)

    emotions = ['calm', 'happy', 'sad', 'angry', 'fear', 'suprise']
    if(not exists('model.pkl')):
        print(run("Data/dataML.csv", "../Datasets/DREAMER.mat"))

    model = joblib.load('model.pkl')
    if(exists(test_path)):
        path = test_path
    elif(exists(ecg_path)):
        path = PFecg.PF_ecg(ecg_path)
    else:
        path = PFecg.PF_generated_ecg(heartrate)

    data = pd.read_csv(path)
    features = np.array(data.loc[:, 'ECG_Rate_Mean':'HRV_SampEn'])
    features = np.nan_to_num(features)
    emotion = model.predict(features)
    return emotions[emotion[0]]


print(app(test="data/emotions_signals/angry.csv"))
