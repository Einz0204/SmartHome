
import os
import face_recognition
import numpy as np

WHITELIST_DIR = "face"
FACEDATA_DIR = "facedata"

os.makedirs(FACEDATA_DIR, exist_ok=True)

def generate_facedata(image_path=None, name=None):
    os.makedirs(FACEDATA_DIR, exist_ok=True)

    if image_path and name:
        # ✅ 單張照片模式（從 GUI 拍攝來的）
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            np.save(os.path.join(FACEDATA_DIR, name + ".npy"), encoding)
            print(f" [單張註冊] 儲存 {name} 向量成功")
        else:
            print(f" [單張註冊] 無法從 {image_path} 擷取人臉向量")
    else:
        # ✅ 批次模式（從 whitelist 資料夾讀）
        for filename in os.listdir(WHITELIST_DIR):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(WHITELIST_DIR, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    encoding = encodings[0]
                    np.save(os.path.join(FACEDATA_DIR, name + ".npy"), encoding)
                    print(f" 儲存 {name} 向量成功")
                else:
                    print(f" 無法從 {filename} 擷取人臉向量")


if __name__ == "__main__":
    generate_facedata()
