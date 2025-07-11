import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import threading
import os
import subprocess
import sys
import uuid
from main2 import FaceRecognizer
from changedata import generate_facedata

class FaceRecognitionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        self.start_btn = tk.Button(root, text="▶ Start Recognition", width=25, height=2,
                                   command=self.start_recognition)
        self.start_btn.pack(pady=5)

        self.log_btn = tk.Button(root, text=" View Log", width=25, height=2,
                                 command=self.open_log)
        self.log_btn.pack(pady=5)
        self.new_btn = tk.Button(root, text="➕ New Member", width=25, height=2,
                         command=self.new_member)
        self.new_btn.pack(pady=5)

        self.cap = None
        self.running = False
        self.recognizer = None

    def start_recognition(self):
        if not self.running:
            self.recognizer = FaceRecognizer()
            self.running = True
            self.update_frame()

    def update_frame(self):
        if self.recognizer and not self.recognizer.finished:
            frame, _ = self.recognizer.get_frame()
            if frame is not None:
                # 顯示到 Tkinter
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        elif self.recognizer and self.recognizer.finished:
            self.running = False
            self.recognizer.release()
            return  # 停止更新

        if self.running:
            self.video_label.after(10, self.update_frame)

    def video_loop(self):
        self.cap = cv2.VideoCapture(0)  # 改成你的來源，如果是影片可以放路徑
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # ===== 這裡可以加入臉部辨識的邏輯 =====
            # 例如： frame = main.detect_and_draw(frame)

            # 顯示畫面到 Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # 每幾毫秒更新一次畫面
            self.video_label.after(10)

        self.cap.release()

    def open_log(self):
        log_path = "log.xlsx"
        if os.path.exists(log_path):
            try:
                if os.name == 'nt':
                    os.startfile(log_path)
                elif os.name == 'posix':
                    subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', log_path])
            except Exception as e:
                messagebox.showerror("Error", f"無法打開 log.xlsx\n{e}")
        else:
            messagebox.showinfo("Log Not Found", "尚未產生 log.xlsx 記錄檔")

    def new_member(self):
    # 1. 輸入密碼
        password = simpledialog.askstring("Password", "請輸入管理密碼", show='*')
        if password != "123":
            messagebox.showerror("Error", "密碼錯誤，無法新增新成員")
            return

    # 2. 輸入姓名
        name = simpledialog.askstring("New Member Name", "請輸入新成員姓名")
        if not name:
            messagebox.showwarning("取消", "未輸入姓名，操作中止")
            return

    # 3. 開啟攝影機
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "無法開啟攝影機")
            return

        messagebox.showinfo("準備拍照", "請面對鏡頭，按下 [空白鍵] 拍照，或 [ESC] 取消")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("新增成員 - 按空白鍵拍照", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC 鍵
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == 32:  # 空白鍵拍照
                filename = f"{name}_{uuid.uuid4().hex[:8]}.jpg"
                save_path = os.path.join("new_faces", filename)
                os.makedirs("new_faces", exist_ok=True)
                cv2.imwrite(save_path, frame)
                cap.release()
                cv2.destroyAllWindows()
                messagebox.showinfo("照片已儲存", f"照片儲存為 {save_path}\n開始產生特徵資料...")
                try:
                    generate_facedata(save_path, name)
                    messagebox.showinfo("完成", f"已成功新增成員：{name}")
                except Exception as e:
                    messagebox.showerror("錯誤", f"新增失敗：{e}")
                return

    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()