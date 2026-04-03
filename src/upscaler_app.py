import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
from cv2 import dnn_superres
import os
import json
from tracker_exe import log_app_usage # 데이터베이스 트래커 연동

class AIUpscalerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 화질 개선기 (4배 증폭)")
        self.root.geometry("500x450")
        
        # 앱 실행 로그
        log_app_usage("ai_upscaler_exe", "app_opened")

        self.model_path = "EDSR_x4.pb"
        # self.sr = dnn_superres.DnnSuperResImpl_create()
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.load_model()

        self.selected_file = ""

        # UI 구성
        tk.Label(root, text="흐릿한 사진을 고화질로 4배 확대합니다.", font=("Arial", 12, "bold")).pack(pady=15)

        self.dnd_label = tk.Label(root, text="여기에 이미지 파일을 드래그 앤 드롭하세요.", 
                                 bg="lightgray", width=50, height=8, relief="sunken")
        self.dnd_label.pack(pady=10)
        self.dnd_label.drop_target_register(DND_FILES)
        self.dnd_label.dnd_bind('<<Drop>>', self.handle_drop)

        # 콤보박스 (확대 알고리즘 선택 - 향후 확장용)
        tk.Label(root, text="AI 모델 선택:").pack(pady=5)
        self.model_var = tk.StringVar(value="EDSR (고품질/느림)")
        model_dropdown = tk.OptionMenu(root, self.model_var, "EDSR (고품질/느림)", "FSRCNN (일반/빠름)", command=self.log_option_change)
        model_dropdown.pack()

        self.status_label = tk.Label(root, text="대기 중...", fg="blue")
        self.status_label.pack(pady=20)

        self.start_btn = tk.Button(root, text="고화질 변환 시작", command=self.process_image, 
                                  bg="blue", fg="white", font=("Arial", 11, "bold"), padx=20, pady=10)
        self.start_btn.pack()

    def load_model(self):
        try:
            self.sr.readModel(self.model_path)
            self.sr.setModel("edsr", 4) # EDSR 모델, 4배 확대
        except Exception as e:
            messagebox.showerror("모델 오류", f"모델 파일을 찾을 수 없습니다.\n{self.model_path} 파일이 같은 폴더에 있는지 확인하세요.")

    def log_option_change(self, value):
        # 콤보박스 사용 흔적 추적 (JSON details)
        log_app_usage("ai_upscaler_exe", "option_changed", details={"selected_model": value})

    def handle_drop(self, event):
        path = event.data
        if path.startswith('{'): path = path.strip('{}')
        
        if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            self.selected_file = path
            self.dnd_label.config(text=f"선택됨:\n{os.path.basename(path)}")
            # 드롭 동작 흔적 추적
            log_app_usage("ai_upscaler_exe", "file_dropped", details={"file_extension": os.path.splitext(path)[1]})
        else:
            messagebox.showwarning("경고", "이미지 파일만 지원합니다.")

    def process_image(self):
        if not self.selected_file:
            messagebox.showwarning("경고", "먼저 이미지를 드래그 앤 드롭 하세요.")
            return

        # 버튼 클릭 흔적 및 시작 상태 추적
        log_app_usage("ai_upscaler_exe", "process_started", details={"file_size_kb": os.path.getsize(self.selected_file) // 1024, "model": self.model_var.get()})

        try:
            self.status_label.config(text="AI 딥러닝 연산 중... (수십 초 소요될 수 있습니다)", fg="orange")
            self.root.update()

            # OpenCV로 이미지 읽기
            image = cv2.imread(self.selected_file)
            
            # AI 해상도 증폭 실행
            result = self.sr.upsample(image)

            # 결과 저장
            output_path = os.path.splitext(self.selected_file)[0] + "_4x_upscaled.png"
            cv2.imwrite(output_path, result)

            self.status_label.config(text=f"완료! 저장됨: {os.path.basename(output_path)}", fg="green")
            messagebox.showinfo("성공", "고화질 변환이 완료되었습니다!")
            
            log_app_usage("ai_upscaler_exe", "process_completed", details={"output_file": output_path})

        except Exception as e:
            self.status_label.config(text="오류 발생", fg="red")
            log_app_usage("ai_upscaler_exe", "process_failed", details={"error_message": str(e)})

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = AIUpscalerApp(root)
    root.mainloop()