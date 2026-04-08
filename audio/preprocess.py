import librosa
import numpy as np
import io
import soundfile as sf

def extract_features(wav_bytes: bytes):
    y, sr = sf.read(io.BytesIO(wav_bytes))
    if y.ndim > 1:
        y = y.mean(axis=1)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    mfcc = mfcc.T

    # Normalisasi
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Sesuaikan input model (contoh CNN)
    return mfcc[np.newaxis, ..., np.newaxis]