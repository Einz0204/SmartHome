import cv2
import  time
from face_recognition import build_white_list_embeddings, recognize_face

print("🔍 建立允許通行的 embedding 資料庫...")
database = build_white_list_embeddings()

if not database:
    print("❌ 沒有找到有效的人臉資料，請將照片放入 white/ 資料夾")
    exit()

print("✅ 資料庫載入完成，啟動攝影機辨識中...")
cap = cv2.VideoCapture(0)
data=0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    name, dist = recognize_face(frame, database)
    if name == "Unknown":
        label = "❌ 拒絕進入"
        color = (0, 0, 255)
    elif name == "No face":
        label = "找不到人臉"
        color = (128, 128, 128)
    else:
        label = f"✅ {name} 允許進入"
        color = (0, 255, 0)
        data=1
        time.sleep(3000) # import time

    cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Access Control System", frame)

    if cv2.waitKey(1) & data==1:
        break

cap.release()
cv2.destroyAllWindows()
