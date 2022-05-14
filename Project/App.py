import numpy as np
import pandas as pd
import joblib
from os.path import exists
import ecg as PFecg


def app(**kwargs):
    test_path = kwargs.get('test', "")
    ecg_path = kwargs.get('ecg', "")
    heartrate = kwargs.get('heartrate', 80)

    emotions = ['calm', 'happy', 'sad', 'angry', 'fear', 'suprise']
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


print(app(ecg="../Datasets/ecg_100hz.csv"))
