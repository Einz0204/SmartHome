import cv2
import time
from face_recognition import build_white_list_embeddings, recognize_face
import os
import datetime
import openpyxl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
white_folder = os.path.join(BASE_DIR, 'white')
history_folder = os.path.join(BASE_DIR, 'history')
history_file = os.path.join(history_folder, 'history.xlsx')

# 建立 history 資料夾
os.makedirs(history_folder, exist_ok=True)

# 初始化資料庫
print("Create an embedding database...")
database = build_white_list_embeddings()

if not database:
    print("Can't find any valid face data.")
    exit()

print("Data loaded successfully. Starting detection system...")
cap = cv2.VideoCapture(0)
access_granted_time = None
countdown_seconds = 3
last_logged_name = None

def log_to_excel(name):
    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M')

    # 如果檔案不存在，就建立新檔並寫入欄位標題
    if not os.path.exists(history_file):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'Access History'
        ws.append(['Date', 'Time', 'Name'])  # 標題列
    else:
        wb = openpyxl.load_workbook(history_file)
        ws = wb.active

    # 寫入一筆新紀錄
    ws.append([date_str, time_str, name])
    wb.save(history_file)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    name, dist = recognize_face(frame, database)

    if name == "Unknown":
        label = "Refuse to entry"
        color = (0, 0, 255)
        access_granted_time = None
        last_logged_name = None

    #儲存截圖
        unknown_folder = os.path.join(history_folder, 'unknown')
        os.makedirs(unknown_folder, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = os.path.join(unknown_folder, f'unknown_{timestamp}.jpg')
        cv2.imwrite(image_path, frame)
    elif name == "No face":
        label = "Can't detect your face"
        color = (128, 128, 128)
        access_granted_time = None
        last_logged_name = None
    else:
         if access_granted_time is None:
            access_granted_time = time.time()  # 設定起始時間
         elapsed_time = time.time() - access_granted_time  # 計算經過時間
         label = f"{name} welcome back ({elapsed_time:.1f}s)"
         color = (0, 255, 0)

    # 顯示標籤
    cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 正數計時
    if access_granted_time:
        elapsed = int(time.time() - access_granted_time)
        cv2.putText(frame, f"Time passed: {elapsed}s", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if access_granted_time and time.time() - access_granted_time >= 3.1:
            print(f"{name} detection completed, turning off the system.")
            break

    cv2.imshow("Access Control System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
