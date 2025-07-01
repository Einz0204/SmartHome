import gdown
import os

model_url = 'https://drive.google.com/uc?id=1K0OgH2XJ4eG5x5pM9fZ7ZsKsHT33d4qX'
output_path = os.path.join("models", "facenet_keras.h5")

if not os.path.exists("models"):
    os.makedirs("models")

print("[INFO] 正在下載 facenet_keras.h5 模型...")
gdown.download(model_url, output_path, quiet=False)
print("[INFO] 模型下載完成！")