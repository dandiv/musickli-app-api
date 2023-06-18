import keras
from keras.models import load_model
from preprocessing.custom_acc import custom_accuracy

class ModelLoader:
    def __init__(self):
        keras.utils.get_custom_objects().update({'custom_accuracy': custom_accuracy})
        self.model = load_model("./assets/model/musickli_model.h5")

    def predict_file(self, audio_file):
        result = self.model.predict(audio_file)
        return result
