import numpy as np
import pandas as pd
import joblib


def run(path: str):
    emotions = ['calm', 'happy', 'sad', 'angry', 'fear', 'suprise']
    model = joblib.load('model.pkl')
    test = pd.read_csv(path)
    tester = np.array(test.loc[:, 'ECG_Rate_Mean':'HRV_SampEn'])
    emotion = model.predict(tester)
    return emotions[emotion[0]]


print(run('test_data/angry.csv'))