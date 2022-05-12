import model
from os.path import exists
import joblib
import processing_Fextraction_DREAMER as dreamer


def App(train_path: str, dreamer_path: str):

    if(not exists(train_path)):
        print("Trainning Data not found, Will generate new data")
        if(exists(dreamer_path)):
            dreamer.PE_dreamer(dreamer_path)
        else:
            return "DREAMER Dataset Not Found"

    model_saved, result = model.supervised_model(train_path)
    joblib.dump(model_saved, 'model.pkl')
    return result


print(App("Data/dataML_Modified.csv", "../Datasets/DREAMER.mat"))
