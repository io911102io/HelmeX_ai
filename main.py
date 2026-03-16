import tkinter as tk
import cv2
import threading
import os
import shutil
import subprocess

from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import pandas as pd

import config
from YOLO_mix import yolo_mix # Import the reset function

# 解決 OMP: Error #15: Initializing libiomp5md.dll... 的問題
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class HelmetDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(config.TK_TITLE)
        self.geometry(config.TK_GEOMETRY)
        # self.state("zoomed") # 預設為最大化，註解此行則為視窗化
        self.stop_event = threading.Event()
        self.stop_event.clear()
        self.pause_event = threading.Event()
        self.pause_event.clear()
        self.detection_running = False        
        self.last_shown_lic_img = None
        self.yolo_model = self.load_yolo_model() # 載入初始模型
        self.model_lock = threading.Lock() # 為模型存取新增一個鎖
        self.server_process = None
        
        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # --- 使用 subprocess 啟動 http.server ---
        print("正在啟動 http.server...")
        self.server_process = subprocess.Popen(
            # 讓 http.server 從 output 資料夾提供檔案
            ["python", "-m", "http.server", "8000", "--directory", "output"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("http.server 已在背景啟動，網址為 http://127.0.0.1:8000")
        
        # tk加入主題
        ttk.Style().theme_use('clam') 
        self._setup_style()

#====================模型載入====================
    def load_yolo_model(self):
        """根據 config 中的路徑載入 YOLO 模型。"""
        if not os.path.exists(config.MODEL_PATH):
            print(f"錯誤：找不到模型檔案 at {config.MODEL_PATH}")
            return None
        print(f"正在載入模型: {config.MODEL_PATH}")
        model = YOLO(config.MODEL_PATH)
        return model

#====================樣式設定====================

    def _setup_style(self):
            style = ttk.Style()
            style.theme_use("clam")  # 設定主題

            # 設定按鈕基本樣式
            style.configure(
                "Custom.TButton",
                background="#282E35",
                foreground="#FFFFFF",
                padding=10,
            )

            # 設定互動反饋效果
            style.map(
                "Custom.TButton",
                background=[
                    ("active", "#63B3FF"),   # hover
                    ("pressed", "#1C7AD9")   # 按下
                ],
                relief=[
                    ("pressed", "sunken")
                ]
            )

#====================主畫面設定====================
    def _create_widgets(self):
# ========== 左側：影片區塊 ==========
        self.rowconfigure(0, weight=2, uniform='r')
        self.rowconfigure(1, weight=1, uniform='r')
        self.columnconfigure(0, weight=2, uniform='c')
        self.columnconfigure(1, weight=1, uniform='c')

        frame_video = tk.Frame(self, bg="#333")
        frame_video.grid(row=0, column=0, padx=10, pady=10, sticky="news")
        frame_video.grid_propagate(False)

        self.label_video = tk.Canvas(frame_video, bg="#333", highlightthickness=0)
        self.label_video.pack(fill="both", expand=True)
        self.canvas_image_item_large = self.label_video.create_image(0, 0, anchor=tk.NW)

        # 左下兩個元件的容器
        self.container = tk.Frame(self)
        self.container.grid(row=1, column=0, sticky='news')
        self.container.columnconfigure(0, weight=1, uniform='c')
        self.container.columnconfigure(1, weight=1, uniform='c')
        self.container.rowconfigure(0, weight=1, uniform='r')

# ========== 左下：影片資訊 ==========
        frame_info = tk.LabelFrame(self.container, text="影片資訊")
        frame_info.grid(row=0, column=0, padx=10, pady=10, sticky="news")
        self.label_name = tk.Label(frame_info, text="影片檔名： ")
        self.label_name.pack(anchor='w')

# ========== 左下：下拉選單 ==========
        combobox_frame = ttk.Frame(self)
        combobox_frame.grid(row=2, column=0, padx=10, pady=10, sticky='w')

        tk.Label(combobox_frame, text="即時攝影機：", font=('Microsoft JhengHei', 12)).pack(side='left')

        self.video_source_combobox = ttk.Combobox(combobox_frame, values=['選擇地段', '板橋區(中山路+漢生東路)','板橋區(中山路+三民路)',
                                                                          '板橋區(環河西路四段)', '板橋區(文化路二段)'], font=('Microsoft JhengHei', 12))
        self.video_source_combobox.current(0)
        self.video_source_combobox.pack(side='left')
        self.video_source_combobox.bind("<<ComboboxSelected>>", self.on_video_source_selected)

# ========== 其他資訊 ========== # 改canvas + scrollbar
        self.extra_info = tk.LabelFrame(self.container,text='偵測未戴安全帽')
        self.extra_info.grid(row=0, column=1, padx=10, pady=10, sticky="news")

        # --- 建立 Canvas 和 Scrollbar ---
        info_canvas = tk.Canvas(self.extra_info, borderwidth=0)

        info_scrollbar = ttk.Scrollbar(self.extra_info, orient="vertical", command=info_canvas.yview)
        info_scrollbar.pack(side="right", fill="y")

        info_canvas.pack(side="left", fill="both", expand=True)
        info_canvas.configure(yscrollcommand=info_scrollbar.set)

        # --- 建立可捲動的 Frame ---
        info_scrollable_frame = tk.Frame(info_canvas)
        info_canvas.create_window((0, 0), window=info_scrollable_frame, anchor='nw')
        info_scrollable_frame.bind("<Configure>", lambda e: info_canvas.configure(scrollregion=info_canvas.bbox("all")))

        # --- 將 Label 放入可滾動的 Frame 中 ---
        self.detected_info_label = tk.Label(info_scrollable_frame, text="等待偵測資料...", justify='left', wraplength=250)
        self.detected_info_label.pack(anchor="nw", fill="x", expand=True, padx=5, pady=5)

# ========== 右側：未戴安全帽的截圖列表 ==========
        frame_result = tk.LabelFrame(self, text="最後違規截圖")
        frame_result.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="news")

        self.canvas_results = tk.Canvas(frame_result)
        self.canvas_results.pack(fill="both", expand=True)

# ========== 右下：按鈕 ==========
        frame_buttons = ttk.Frame(self)
        frame_buttons.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="se")
        btn_select_video = ttk.Button(frame_buttons, text="選擇影片", style="Custom.TButton", command=self.select_video_file)
        btn_select_video.grid(row=0, column=0, padx=5)
        self.btn_start_detection = ttk.Button(frame_buttons, text="開始偵測", style="Custom.TButton", command=self.start_detection)
        self.btn_start_detection.grid(row=0, column=1, padx=5)
        btn_export = ttk.Button(frame_buttons, text="違規紀錄", style="Custom.TButton", command=self.open_new_window)
        btn_export.grid(row=0, column=2, padx=5)
        btn_clear = ttk.Button(frame_buttons, text="清除結果", style="Custom.TButton", command=self.clear_results)
        btn_clear.grid(row=0, column=3, padx=5)
        btn_setting = ttk.Button(frame_buttons, text="設定", style="Custom.TButton")
        btn_setting.grid(row=0, column=4, padx=5)
        btn_clear = ttk.Button(frame_buttons, text="清除結果", style="Custom.TButton", command=self.clear_results) #ju
        btn_clear.grid(row=0, column=3, padx=5) #ju
        btn_setting = ttk.Button(frame_buttons, text="設定", style="Custom.TButton", command=self.open_settings_window) #ju
        btn_setting.grid(row=0, column=4, padx=5) #ju
        
# ========== 資料視窗 ===========
    def open_new_window(self):
        df = pd.read_csv(config.CSV)
        FONT_TITLE = ('Arial', 24)
        new_window = tk.Toplevel(self)
        new_window.title("違規紀錄")
        new_window.geometry(config.NEW_GEOMETRY)

        # --- 建立 Canvas 和 Scrollbar ---
        main_frame = tk.Frame(new_window)
        main_frame.pack(fill='both', expand=True)

        canvas = tk.Canvas(main_frame)
        canvas.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')

        canvas.configure(yscrollcommand=scrollbar.set)

        # --- 建立可捲動的 Frame ---
        scrollable_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

        # 當 scrollable_frame 的大小改變時，更新 canvas 的 scrollregion
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # --- 綁定滑鼠滾輪事件 ---
        def _on_mouse_wheel(event):
            # 根據滾輪滾動的方向和單位來捲動 Canvas
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mouse_wheel)

        tk.Button(new_window, text="關閉視窗", command=new_window.destroy).pack(side='bottom', pady=10)

        # 使用 config.df 來顯示資料
        df = pd.read_csv(config.CSV)
        if df.empty:
            tk.Label(scrollable_frame, text="目前沒有任何違規紀錄。").pack(pady=20)
        else:
            for i, row in df.iterrows():
                frame = tk.LabelFrame(scrollable_frame, text=f'第 {i+1} 筆資料')
                frame.pack(padx=10, pady=10, fill='x', expand=True)
                photo = ImageTk.PhotoImage(Image.open(row['image_url']).resize((300, 500)))
                tk.Label(frame, text=f"車牌號碼: {row['kar_id']}", font=FONT_TITLE).pack(pady=5)
                driver_photo_label = tk.Label(frame, image=photo)
                driver_photo_label.image = photo # 將圖片物件綁定到 Label 上，防止被回收
                driver_photo_label.pack(pady=5)

#=================Button_設定====================
    def open_settings_window(self):
        """開啟一個新視窗來調整 YOLO 參數。"""
        settings_window = tk.Toplevel(self)
        settings_window.title("調整 YOLO 參數")
        settings_window.geometry("350x380")
        settings_window.transient(self) # 讓設定視窗保持在主視窗之上
        settings_window.grab_set()      # 獨佔焦點

        # --- 即時更新函式 ---
        def update_conf(value):
            """即時更新信心度"""
            config.CONF = float(value)
            print(f"信心度 (CONF) 即時更新為: {config.CONF:.2f}")

        def update_iou(value):
            """即時更新重疊度"""
            config.IOU = float(value)
            print(f"重疊度 (IOU) 即時更新為: {config.IOU:.2f}")

        def update_tracker(event):
            """即時更新追蹤器"""
            config.TRACKER_TYPE = tracker_combobox.get()
            print(f"追蹤器 (Tracker) 即時更新為: {config.TRACKER_TYPE}")

        def update_model(event):
            """即時更新並重新載入模型"""
            selected_model_file = model_combobox.get()
            new_model_path = os.path.join(models_dir, selected_model_file)
            if new_model_path != config.MODEL_PATH:
                config.MODEL_PATH = new_model_path
                with self.model_lock: # 鎖定以安全地替換模型
                    self.yolo_model = self.load_yolo_model()

        # --- CONF (信心度) ---
        tk.Label(settings_window, text="信心度 (Confidence)", font=('Microsoft JhengHei', 10)).pack(pady=(10, 0))
        conf_frame = tk.Frame(settings_window)
        conf_frame.pack(pady=(0, 10))
        
        conf_slider = tk.Scale(conf_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=250, command=update_conf)
        conf_slider.set(config.CONF)
        conf_slider.pack(side=tk.LEFT)

        # --- IOU (重疊度) ---
        tk.Label(settings_window, text="重疊度 (IOU)", font=('Microsoft JhengHei', 10)).pack(pady=(10, 0))
        iou_frame = tk.Frame(settings_window)
        iou_frame.pack(pady=(0, 10))

        iou_slider = tk.Scale(iou_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=250, command=update_iou)
        iou_slider.set(config.IOU)
        iou_slider.pack(side=tk.LEFT)

        # --- Tracker Type (追蹤器類型) ---
        tk.Label(settings_window, text="追蹤器類型 (Tracker Type)", font=('Microsoft JhengHei', 10)).pack(pady=(10, 0))
        tracker_frame = tk.Frame(settings_window)
        tracker_frame.pack(pady=(0, 10))

        tracker_options = ['botsort.yaml', 'bytetrack.yaml']
        tracker_combobox = ttk.Combobox(tracker_frame, values=tracker_options, state="readonly", width=38)
        try:
            tracker_combobox.set(config.TRACKER_TYPE)
        except tk.TclError: # 如果 config 的值不在選項中，設定預設值
            tracker_combobox.set(tracker_options[0])
        tracker_combobox.pack(side=tk.LEFT)
        tracker_combobox.bind("<<ComboboxSelected>>", update_tracker)

        # --- Model (模型選擇) ---
        tk.Label(settings_window, text="模型選擇 (Model Selection)", font=('Microsoft JhengHei', 10)).pack(pady=(10, 0))
        model_frame = tk.Frame(settings_window)
        model_frame.pack(pady=(0, 10))

        # 掃描 models 資料夾取得 .pt 檔案列表
        models_dir = 'models'
        if os.path.exists(models_dir):
            model_options = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        else:
            model_options = [os.path.basename(config.MODEL_PATH)] # 如果資料夾不存在，至少顯示當前模型

        model_combobox = ttk.Combobox(model_frame, values=model_options, state="readonly", width=38)
        current_model_filename = os.path.basename(config.MODEL_PATH)
        if current_model_filename in model_options:
            model_combobox.set(current_model_filename)
        elif model_options:
            model_combobox.set(model_options[0])
        model_combobox.pack(side=tk.LEFT)
        model_combobox.bind("<<ComboboxSelected>>", update_model)

        # --- 按鈕 ---
        close_button = ttk.Button(settings_window, text="關閉", style="Custom.TButton", command=settings_window.destroy)
        close_button.pack(pady=10)


#=================Button_選影片====================

    def select_video_file(self):
        self.pause_event.clear()
        file_path = filedialog.askopenfilename(
            title="請選擇影片檔案",
            filetypes=[("影片檔案", "*.mp4 *.avi *.mov"), ("所有檔案", "*.*")]
        )

        if not file_path:
            return #沒上傳直接取消
        
        self.video_path = file_path
        config.VIDEO = file_path   # <--- 把路徑寫回 config

        self.is_running = False
        self.stop_event.clear()

        filename = os.path.basename(file_path)
        config.FILENAME = filename
        self.label_name.config(text=f"影片檔名：{filename}")
        print(f"已選擇影片: {self.video_path}")

        # 如果已經在偵測中，選擇新影片後應該停止並重置按鈕狀態
        if self.detection_running:
            self.clear_results()
            config.VIDEO = file_path
            config.FILENAME = filename
            self.label_name.config(text=f"影片檔名：{filename}")
        self.start_detection()

    def on_video_source_selected(self, event):
        """當 Combobox 選項被選擇時觸發。"""
        selected_option = self.video_source_combobox.get()
        video_path = None
        filename = ""        
        if self.detection_running:
            self.clear_results()

        if selected_option == '選擇地段':
            pass

        elif selected_option == '板橋區(中山路+漢生東路)':
            video_path = 'https://cctvatis1.ntpc.gov.tw/hls/C000012/live.m3u8'
            filename = "板橋區即時影像(中山路+漢生東路)"

        elif selected_option == '板橋區(中山路+三民路)':
            video_path = 'https://cctvatis1.ntpc.gov.tw/hls/C000022/live.m3u8'
            filename = "板橋區即時影像(中山路+三民路)"

        elif selected_option == '板橋區(環河西路四段)':
            video_path = 'https://cctvatis4.ntpc.gov.tw/hls/C000400/live.m3u8'
            filename = "板橋區即時影像(環河西路四段)"

        elif selected_option == '板橋區(文化路二段)':
            video_path = 'https://cctvatis6.ntpc.gov.tw/hls/C000118/live.m3u8'
            filename = "板橋區即時影像(文化路二段)"

        if video_path:
            config.VIDEO = video_path
            config.FILENAME = filename
            self.video_path = video_path
            self.label_name.config(text=f"影片檔名：{filename}")
            self.start_detection()

#=================Button_開始偵測====================

    def start_detection(self):
        video_path = config.VIDEO

        if not video_path:
            print("錯誤：尚未選擇任何影片檔案。")
            return
        
        if not self.detection_running:
            print(f"從選擇影片開始偵測: {video_path}")
            self.detection_running = True
            self.stop_event.clear()
            self.pause_event.clear()  # 開始時不暫停
            self.start_thread(video_path)
            self.btn_start_detection.config(text="開始偵測")

        elif self.pause_event.is_set():
            # 從暫停狀態恢復
            print("繼續偵測")
            self.pause_event.clear()
            self.btn_start_detection.config(text="繼續偵測")
        else:
            # 暫停偵測
            print("暫停偵測")
            self.pause_event.set()
            self.btn_start_detection.config(text="暫停偵測")

#====================Button_清除資料====================

    def clear_results(self):
        """
        停止影片播放和偵測，清除所有已儲存的資料，並重置 GUI 顯示。
        """
        # 1. 停止所有正在執行的偵測執行緒
        print("清除結果：停止偵測執行緒...")
        self.stop_event.set()
        self.pause_event.set() # 確保任何等待中的執行緒被釋放

        # 給予短暫時間讓執行緒能夠響應停止事件
        # 這裡假設 update_video_feed 是在一個名為 self.video_thread 的執行緒中運行
        if hasattr(self, "video_thread") and self.video_thread.is_alive():
            self.video_thread.join(timeout=1) # 等待執行緒結束，設定超時時間
            if self.video_thread.is_alive():
                print("警告：影片處理執行緒未能及時停止。")

        # 2. 執行 reset.py 中的清除環境功能
        print("清除結果：重置環境...")
        self.reset_environment()

        # 3. 重置 GUI 顯示
        print("清除結果：重置 GUI 顯示...")
        self.label_video.delete("all") # 清除影片顯示區域        
        self.canvas_image_item_large = self.label_video.create_image(0, 0, anchor=tk.NW) # 立即重建 image item
        self.detected_info_label.config(text="等待偵測資料...") # 重置下方資訊
        for widget in self.canvas_results.winfo_children(): # 清除右側圖片
            widget.destroy()
        self.label_name.config(text="影片檔名： ") # 重置影片檔名顯示
        self.btn_start_detection.config(text="開始偵測") # 重置按鈕文字

        # 4. 重置影片路徑和狀態變數
        self.video_path = None
        config.VIDEO = None
        config.FILENAME = None
        self.stop_event.set()
        self.pause_event.set()
        self.detection_running = False

#====================背景執行====================
    def start_thread(self, video_path):
        self.stop_event.clear()
        # 啟動影片-執行緒
        video_thread = threading.Thread(target=self.update_video_feed, args=(video_path,))
        video_thread.daemon = True  # 設置為守護執行緒，主程式退出時會跟著退出
        video_thread.start()
        self.video_thread = video_thread
        
    def add_new_trace_and_update(self, trace):
        """在主執行緒中安全地更新 CSV 和 GUI。"""
        df = pd.read_csv(config.CSV)
        new_df = pd.concat([df, pd.DataFrame([trace])], ignore_index=True)
        new_df.to_csv(config.CSV, index=False, encoding='utf-8-sig')
        
        # 直接呼叫更新方法
        self.info_update()
        self.img_update()
        return new_df

    # ===更新下方資訊===
    def info_update(self):
        text = ''
        df = pd.read_csv(config.CSV)
        for _, r in df.iterrows():
            text += f"車牌：{r['kar_id']}，時間：{r['date']}, 安全帽：{r['helmet']}\n"
        self.detected_info_label.config(text=text)

    # ===更新右側圖片===
    def img_update(self):
        """
        透過檔名配對，更新右側顯示的最新一筆駕駛員與車牌截圖。
        """

        # 1. 在 output 資料夾中，只尋找包含 'lic' 的最新車牌截圖  #ju
        lic_files = [f for f in os.listdir(config.OUTPUT) if config.LIC_NAME in f and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not lic_files:
            return

        latest_lic_file = max(lic_files, key=lambda f: os.path.getmtime(os.path.join(config.OUTPUT, f)))

        # 2. 如果最新的車牌圖片和上次顯示的一樣，就跳過，避免不必要的刷新 #ju
        if latest_lic_file == self.last_shown_lic_img:
            return
        
        # 3. 根據新的命名規則，從車牌檔名推導出對應的駕駛員檔名 #ju
        # 例如: 'frame123_big0_lic.jpg' -> 'frame123_big0_driver.jpg'
        original_driver_filename = latest_lic_file.replace(config.LIC_NAME, config.DRIVER_NAME)
        driver_img_path = os.path.join(config.OUTPUT, original_driver_filename)
        lic_img_path = os.path.join(config.OUTPUT, latest_lic_file)

        # 4. 檢查對應的駕駛員圖片是否存在
        if os.path.exists(driver_img_path):
            # 清除舊的圖片 (必須在主執行緒中操作 GUI) #ju
            for widget in self.canvas_results.winfo_children():
                widget.destroy()

            # 顯示新的配對圖片
            # 調整 driver 圖片大小以符合畫布，車牌使用較小尺寸
            driver_img = Image.open(driver_img_path).resize((300, 400), Image.Resampling.LANCZOS)
            lic_img = Image.open(lic_img_path).resize((150, 50), Image.Resampling.LANCZOS)

            self.tk_driver_img = ImageTk.PhotoImage(driver_img)
            self.tk_lic_img = ImageTk.PhotoImage(lic_img)

            tk.Label(self.canvas_results, image=self.tk_driver_img).pack()
            tk.Label(self.canvas_results, image=self.tk_lic_img).pack()

            self.last_shown_lic_img = latest_lic_file # 更新已顯示的圖片紀錄
    
    # === 更新影片 ===
    def update_video_feed(self, video_path):
        """在背景執行緒中執行，從 yolo_driver 獲取影像並更新 GUI。"""
        try: # 確保在循環內部檢查停止事件
            # 將 stop_event 和 app instance (self) 傳遞給 yolo_mix
            MAX_CHANCES = 2
            chance = MAX_CHANCES
            while chance > 0:
                frame_generator = yolo_mix(self.stop_event, video_path=video_path, app_instance=self)
                for frame, has_new_detection in frame_generator:
                    # 在更新畫面之前，再次檢查停止事件
                    if self.stop_event.is_set():
                        break

                    # 成功讀取, chance回滿血
                    chance = MAX_CHANCES

                    # 如果 yolo_mix 偵測到新違規，它會在這裡觸發 GUI 更新
                    # 我們在 add_new_trace_and_update 中處理了更新，所以這裡不需要再做什麼
                    # 只需要持續更新影片畫面即可

                    # 動態獲取 Canvas 的當前大小
                    canvas_width = self.label_video.winfo_width()
                    canvas_height = self.label_video.winfo_height()
                    frame_large = cv2.resize(frame, (canvas_width, canvas_height))
                    frame_large_rgb = cv2.cvtColor(frame_large, cv2.COLOR_BGR2RGB)

                    self.photo_large = ImageTk.PhotoImage(image=Image.fromarray(frame_large_rgb))
                    if self.canvas_image_item_large is None:
                        self.canvas_image_item_large = self.label_video.create_image(0, 0, anchor=tk.NW)
                    self.label_video.itemconfig(self.canvas_image_item_large, image=self.photo_large)
                    self.label_video.image = self.photo_large
                    # 檢查是否需要暫停
                    self.pause_event.wait() # 如果 pause_event 被設置，則會在此處阻塞
                
                if self.stop_event.is_set():
                    chance -= 1
                    print("影片處理執行緒已停止。")
                else: 
                    chance -= 1
                    print(f"影片處理失敗，還有 {chance} 次機會。一秒後重試")
                    cv2.waitKey(1000)
            

        except Exception as e:
            print(f"影片處理時發生錯誤: {e}")
        finally:
            # 確保不論執行緒是正常結束還是發生錯誤，都會將執行狀態設為 False
            self.detection_running = False            
            print("影片處理執行緒已停止。")

    def on_closing(self):
        """在關閉視窗時被呼叫，用來停止背景執行緒。"""
        print("正在關閉應用程式...")
        self.stop_event.set()
        self.pause_event.set()
        self.destroy()
        # --- 關閉 http.server process ---
        if self.server_process:
            print("正在關閉 http.server...")
            self.server_process.terminate()
            self.server_process.wait()
        self.reset_environment()

    def reset_environment(self):
        """
        刪除並重新建立所有由 YOLO 流程產生的資料夾，並重置 CSV 紀錄檔。
        """
        # 從 config.py 讀取需要重置的資料夾路徑列表
        # 我們也加入了 PROCESSING_FOLDER ('imgs')，因為它也是暫存區
        folder = config.OUTPUT

        print("--- 開始重置環境 ---")

        # 1. 處理資料夾
        try:
            if os.path.exists(folder):                
                print(f"正在刪除資料夾: {folder}")
                shutil.rmtree(folder) # 遞迴刪除整個資料夾
            print(f"正在建立空資料夾: {folder}")
            os.makedirs(folder) # 重新建立空資料夾
        except Exception as e:
            print(f"處理資料夾 {folder} 時發生錯誤: {e}")

        # 2. 重置 CSV 檔案
        print(f"正在重置紀錄檔: {config.CSV}")
        df_empty = pd.DataFrame(columns=['video_name', 'kar_id', 'date', 'helmet', 'image_url'])
        df_empty.to_csv(config.CSV, index=False, encoding='utf-8-sig')
        self.last_shown_lic_img = None

        print("--- 環境重置完成 ---")
#======================================================================
if __name__ == "__main__":
    app = HelmetDetectionApp()
    app.mainloop()