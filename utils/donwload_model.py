import os
import gdown

MODEL_PATH = "models/CRNN_Model.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        
        url = "https://drive.google.com/file/d/1pBzXZcZrtun_wkIeavfgxxbMxG4okZ6J"
        
        os.makedirs("models", exist_ok=True)
        gdown.download(url, MODEL_PATH, quiet=False)
        
        print("Model downloaded!")
    else:
        print("Model already exists.")