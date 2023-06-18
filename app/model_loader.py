import keras
from keras.models import load_model

def custom_accuracy(y_true, y_pred):
    return 0.5;

class ModelLoader:
    def predict_file(self, audio_file):
        keras.utils.get_custom_objects().update({'custom_accuracy': custom_accuracy})
        model = load_model("./assets/model/musickli_model.h5")
        result = model.predict(audio_file)
        return result
