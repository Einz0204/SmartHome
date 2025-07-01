import cv2
import time
import os
import datetime
import openpyxl
from face_recognition import build_white_list_embeddings, recognize_face

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
white_folder = os.path.join(BASE_DIR, 'white')
history_folder = os.path.join(BASE_DIR, 'history')
history_file = os.path.join(history_folder, 'history.xlsx')

# 建立 history 資料夾
os.makedirs(history_folder, exist_ok=True)

# 初始化資料庫
print("🔍 建立白名單資料庫...")
database = build_white_list_embeddings()
if not database:
    print("❌ 沒有找到任何白名單人臉")
    exit()

print("✅ 資料庫載入成功，啟動系統...")
cap = cv2.VideoCapture(0)

access_granted_time = None
countdown_seconds = 3
recognized_name = None  # 紀錄誰被辨識
unknown_triggered = False  # 控制 unknown 只執行一次截圖

def log_to_excel(name):
    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M')
    os.makedirs(history_folder, exist_ok=True)

    if not os.path.exists(history_file):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'Access History'
        ws.append(['Date', 'Time', 'Name'])
    else:
        wb = openpyxl.load_workbook(history_file)
        ws = wb.active

    ws.append([date_str, time_str, name])
    wb.save(history_file)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    name, dist = recognize_face(frame, database)

    if name != recognized_name:
        recognized_name = name
        access_granted_time = time.time() if name not in ["No face"] else None
        unknown_triggered = False

    if name == "Unknown":
        label = "❌ Refuse to entry (Exit in 3s)"
        color = (0, 0, 255)
        if access_granted_time is None:
            access_granted_time = time.time()
            recognized_name = "Unknown"
        elapsed = int(time.time() - access_granted_time)
        countdown = max(0, countdown_seconds - elapsed)
        label += f" - {countdown}s"

        # 截圖僅一次
        if elapsed >= countdown_seconds:
            if recognized_name == "Unknown":
                # 在最後一刻截圖
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                unknown_folder = os.path.join(history_folder, 'unknown')
                os.makedirs(unknown_folder, exist_ok=True)
                path = os.path.join(unknown_folder, f'unknown_{timestamp}.jpg')
                cv2.imwrite(path, frame)
                print(f"📸 Unknown 截圖儲存於 {path}")

    elif name == "No face":
        label = "😐 Can't detect face"
        color = (128, 128, 128)
        access_granted_time = None
        recognized_name = None
        unknown_triggered = False

    else:
        if access_granted_time is None:
            access_granted_time = time.time()
            recognized_name = name
        elapsed = time.time() - access_granted_time
        label = f"✅ {name} welcome ({elapsed:.1f}s)"
        color = (0, 255, 0)

    # 顯示文字與時間
    cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if access_granted_time:
        elapsed = time.time() - access_granted_time
        cv2.putText(frame, f"Timer: {elapsed:.1f}s", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 經過倒數時間，進行紀錄與關閉
        if elapsed >= countdown_seconds:
            if recognized_name == "Unknown":
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                unknown_folder = os.path.join(history_folder, 'unknown')
                os.makedirs(unknown_folder, exist_ok=True)
                path = os.path.join(unknown_folder, f'unknown_{timestamp}.jpg')
                cv2.imwrite(path, frame)
                print(f"📸 Unknown 截圖儲存於 {path}")

            log_to_excel(recognized_name)
            print(f"📒 已記錄訪問：{recognized_name}")
            print("🔚 系統自動關閉")
            break

    cv2.imshow("Access Control System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 手動關閉")
        break

cap.release()
cv2.destroyAllWindows()
