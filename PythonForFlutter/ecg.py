import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt


def plot_settings():
    plt.rcParams['figure.figsize'] = [15, 9]
    plt.rcParams['font.size'] = 13


def customiz_data(data):
    data = pd.DataFrame(data, columns=[
        'ECG_Rate_Mean', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_HTI', 'HRV_HF', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S', 'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_DFA_alpha1', 'HRV_DFA_alpha1_ExpRange', 'HRV_DFA_alpha1_ExpMean', 'HRV_DFA_alpha1_DimRange', 'HRV_DFA_alpha1_DimMean', 'HRV_ApEn', 'HRV_SampEn'
    ])
    return data


def processing(ecg_signal):
    ecg_cleaned = nk.ecg_clean(
        ecg_signal, sampling_rate=256)
    instant_peaks, rpeaks, = nk.ecg_peaks(ecg_cleaned, sampling_rate=256)
    rate = nk.ecg_rate(rpeaks, sampling_rate=256,
                       desired_length=len(ecg_cleaned))
    quality = nk.ecg_quality(ecg_cleaned, sampling_rate=256)

    signals = pd.DataFrame({"ECG_Raw": ecg_signal,
                            "ECG_Clean": ecg_cleaned,
                            "ECG_Rate": rate,
                            "ECG_Quality": quality})
    signals = pd.concat([signals, instant_peaks], axis=1)
    info = rpeaks

    return signals, info


def PF_generated_ecg(heartRate):
    # plot_settings()
    ecg = nk.ecg_simulate(sampling_rate=256, heart_rate=heartRate)
    # nk.signal_plot(ecg)
    signal, info = processing(ecg)
    #nk.ecg_plot(signal[:3000], sampling_rate=256)
    data = nk.ecg_intervalrelated(signal)
    data = customiz_data(data)
    data.to_csv("generated_ECG_256hz.csv")
    return "generated_ECG_256hz.csv"


def PF_ecg(path):
    # plot_settings()
    ecg = pd.read_csv(path)
    # nk.signal_plot(ecg['ECG'])
    signal, info = nk.ecg_process(ecg["ECG"], sampling_rate=100)
    #nk.ecg_plot(signal[:3000], sampling_rate=100)
    data = nk.ecg_intervalrelated(signal)
    data = customiz_data(data)
    data.to_csv("PF_ECG_100hz.csv")
    return "PF_ECG_100hz.csv"
