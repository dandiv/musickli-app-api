from keras.models import load_model


class ModelLoader:
    def __init__(self):
        self.model_file = open("app/assets/model/musickli_model.h5", "r")

    def predict_file(self, audio_file):
        model = load_model(self.model_file)
        result = model.predict(audio_file)
        return result
