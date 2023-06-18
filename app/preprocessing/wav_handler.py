import librosa
import numpy as np

class WavHandler:
    def __init__(self, file_path: str, time_offset: float=0.0, duration: float=1.0):
        self.file_path = file_path
        self.y_audio_time_series, self.sr = librosa.load(file_path, offset=time_offset, duration=duration)
        self.__set_harmonic()
        self.__set_mfcc()
        self.__set_spectrogram()
        self.__set_chroma_energy()
        self.__set_contrast()

    def __set_harmonic(self):
        """
        Determine whether an instrument is harmonic or percussive
        """
        y_harmonic, y_percussive = librosa.effects.hpss(self.y_audio_time_series)
        if np.mean(y_harmonic) > np.mean(y_percussive):
            self.__is_harmonic = True
        self.__is_harmonic = False

    def __set_mfcc(self):
        """
        Mel-frequency cepstral coefficients (MFCCs)
        """
        mfcc_feature = librosa.feature.mfcc(y=self.y_audio_time_series,
                                            sr=self.sr,
                                            n_mfcc=13)
        self.__mfcc_val = np.mean(mfcc_feature, axis=1)

    def __set_spectrogram(self):
        """
        Mel-scaled spectrogram
        """
        spectrogram = librosa.feature.melspectrogram(y=self.y_audio_time_series,
                                                    sr=self.sr,
                                                    n_mels=128,
                                                    fmax=8000)
        self.__spectrogram = np.mean(spectrogram, axis=1)

    def __set_chroma_energy(self):
        """
        Compute temporally average chroma
        """
        chroma = librosa.feature.chroma_cens(y=self.y_audio_time_series,
                                            sr=self.sr)
        self.__chroma = np.mean(chroma, axis=1)

    def __set_contrast(self):
        """
        Compute spectral contrastim
        """
        contrast = librosa.feature.spectral_contrast(y=self.y_audio_time_series,
                                                    sr=self.sr)
        self.__contrast = np.mean(contrast, axis=1)

    def get_harmonic(self) -> bool:
        return self.__is_harmonic

    def get_mfcc(self) -> np.ndarray:
        return self.__mfcc_val

    def get_spectrogram(self) -> np.ndarray:
        return self.__spectrogram

    def get_chroma_energy(self) -> np.ndarray:
        return self.__chroma

    def get_contrast(self) -> np.ndarray:
        return self.__contrast

    def get_features(self):
        return [self.__is_harmonic, 
                self.__mfcc_val, 
                self.__spectrogram, 
                self.__chroma, 
                self.__contrast]
