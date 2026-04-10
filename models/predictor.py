import numpy as np

MODEL_PATH = "models/CRNN_Model.keras"

class SoundModel:
    def predict(self, x):
        return 0, 0.5
    
    def predict(self, features: np.ndarray):
        """
        features shape: (1, time, freq) atau sesuai model kamu
        """
        preds = self.model.predict(features)
        return preds
