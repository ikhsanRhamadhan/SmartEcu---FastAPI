from models.predictor import SoundModel
from audio.convert import webm_to_wav
from audio.preprocess import extract_features

model = SoundModel()  # 🔥 LOAD SEKALI

def run_inference(audio_bytes: bytes):
    wav_bytes = webm_to_wav(audio_bytes)
    features = extract_features(wav_bytes)
    prediction = model.predict(features)

    return prediction