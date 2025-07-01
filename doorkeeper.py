facenet_access_system/
├── main.py                      # 主程式
├── face_recognition.py         # 處理人臉比對與embedding
├── database/
│   ├── alice.jpg
│   ├── bob.jpg
├── models/
│   └── facenet_keras.h5        # 預訓練 FaceNet 模型
├── embeddings/
│   └── embeddings.pkl          # 儲存所有已註冊使用者的embedding
├── requirements.txt