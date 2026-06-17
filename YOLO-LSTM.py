import os
import sys
import json
import joblib
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')  # 避免与 Tkinter 冲突

# ==================== 全局配置 ====================
YOLO_MODEL_PATH = r"E:\pycode\yolov12-main\YOLO-LSTM\exp29\weights\best.pt"  # YOLO 权重
LSTM_MODEL_DIR = "trained_model"            # LSTM 模型文件夹 (与本脚本同目录)
SUPPORTED_IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
IMAGE_DISPLAY_WIDTH = 600                  # 图片显示宽度（像素）
# ===================================================


# -------------------- 工具函数 --------------------
def load_lstm_model(model_dir):
    """加载 LSTM 模型、标准化器和配置"""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"LSTM model folder not found: {model_dir}")
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    feature_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
    target_scaler = joblib.load(os.path.join(model_dir, 'target_scaler.pkl'))
    model = load_model(os.path.join(model_dir, 'model.h5'))
    return model, feature_scaler, target_scaler, config


def calculate_brightness(image):
    """输入 PIL Image，返回平均亮度、最高亮度（0-255）"""
    gray = image.convert('L')
    arr = np.array(gray)
    return np.mean(arr), np.max(arr)


def parse_filename(img_filename):
    """
    从图片文件名中提取 layers 和 power。
    例: "1-1000-2519023 (1).png" -> (1.0, 1000.0)
    """
    name = os.path.splitext(os.path.basename(img_filename))[0]
    parts = name.split('-')
    try:
        layers = float(parts[0])
        power = float(parts[1])
    except (IndexError, ValueError):
        raise ValueError(f"Filename format error, cannot parse layers and power: {img_filename}")
    return layers, power


def extract_detection_features(results):
    """
    从 YOLO 检测结果中提取 MPL（熔池长度）和 MPW（熔池宽度）。
    策略：取置信度最高的检测框，将其 width/height 中较大者作为 MPL，较小者作为 MPW。
    返回 (MPL, MPW) 或 (None, None)
    """
    if results.boxes is None or len(results.boxes) == 0:
        return None, None
    confs = results.boxes.conf.cpu().numpy()
    best_idx = np.argmax(confs)
    xywh = results.boxes.xywh.cpu().numpy()
    _, _, w, h = xywh[best_idx]
    mpl = max(w, h)
    mpw = min(w, h)
    return mpl, mpw


def draw_boxes(image, results):
    """在 PIL Image 上绘制所有检测框（蓝色）以及标签，返回新图"""
    draw = ImageDraw.Draw(image)
    if results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy()
        names = results.names
        for box, class_id in zip(xyxy, cls):
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
            label = names[int(class_id)]
            draw.text((x1, y1 - 15), label, fill="blue")
    return image


# -------------------- 主 GUI 类 --------------------
class IntegratedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Melt Pool Detection + LSTM Prediction")
        self.root.geometry("1000x900")
        self.root.minsize(900, 700)
        self.root.configure(bg='#f0f0f0')   # 浅灰背景

        # 数据状态
        self.img_folder = None
        self.img_files = []            # 排序后的图片文件名列表（仅文件名）
        self.img_abs_paths = []        # 对应的绝对路径
        self.current_idx = 0

        # 处理结果
        self.features = None           # numpy数组 (n,6) 存放所有图片特征
        self.valid_indices = []        # 有效图片在原始序列中的索引
        self.predictions = None        # LSTM 预测输出 (n_valid-look_back, 3)
        self.pred_mapping = {}         # 原索引 -> 预测向量或None

        # 模型
        self.yolo = None
        self.lstm_model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.config = None
        self.look_back = 2             # 将在加载 LSTM 时更新

        # 启动时加载模型
        self.load_models()
        # 创建界面
        self.create_widgets()

    # ---------- 模型加载 ----------
    def load_models(self):
        """加载 YOLO 和 LSTM 模型，若失败则退出"""
        # 加载 YOLO
        try:
            if not os.path.exists(YOLO_MODEL_PATH):
                raise FileNotFoundError(f"YOLO model file not found: {YOLO_MODEL_PATH}")
            self.yolo = YOLO(YOLO_MODEL_PATH, task="detect")
            print("YOLO model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"YOLO model loading failed:\n{e}")
            sys.exit(1)

        # 加载 LSTM
        try:
            self.lstm_model, self.feature_scaler, self.target_scaler, self.config = load_lstm_model(LSTM_MODEL_DIR)
            self.look_back = self.config['look_back']
            print(f"LSTM model loaded successfully (look_back={self.look_back}, "
                  f"input_features={self.config['num_features']}, outputs={self.config['num_outputs']})")
        except Exception as e:
            messagebox.showerror("Error", f"LSTM model loading failed:\n{e}")
            sys.exit(1)

    # ---------- 界面构建 ----------
    def create_widgets(self):
        # 顶部工具栏
        toolbar = tk.Frame(self.root, bg='#e0e0e0', relief=tk.RAISED, bd=1)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)

        btn_select = tk.Button(toolbar, text="Select Image Folder", command=self.select_folder,
                               font=("Segoe UI", 11), bg="#4CAF50", fg="white", padx=10, pady=3)
        btn_select.pack(side=tk.LEFT, padx=10, pady=8)

        self.lbl_folder = tk.Label(toolbar, text="No folder selected", fg="gray",
                                   font=("Segoe UI", 9), bg='#e0e0e0')
        self.lbl_folder.pack(side=tk.LEFT, padx=20)

        # 图片显示 Canvas
        self.canvas = tk.Canvas(self.root, bg='white', width=IMAGE_DISPLAY_WIDTH, height=400,
                                relief=tk.GROOVE, bd=2)
        self.canvas.pack(pady=15)
        self.photo_ref = None  # 防止图片被垃圾回收

        # 信息显示区（左右分栏）
        info_frame = tk.Frame(self.root, bg='#f0f0f0')
        info_frame.pack(fill=tk.BOTH, padx=20, pady=5)

        # 左栏：图片信息、特征
        left_info = tk.Frame(info_frame, bg='#f0f0f0')
        left_info.pack(side=tk.LEFT, anchor=tk.W, padx=10)

        # 使用统一字体
        lbl_font = ("Segoe UI", 10)
        lbl_font_small = ("Segoe UI", 9)
        lbl_font_title = ("Segoe UI", 12, "bold")

        self.lbl_filename = tk.Label(left_info, text="Image: ", font=lbl_font, bg='#f0f0f0')
        self.lbl_filename.grid(row=0, column=0, sticky='w', pady=2)
        self.lbl_layers = tk.Label(left_info, text="Number of layers: ", font=lbl_font, bg='#f0f0f0')
        self.lbl_layers.grid(row=1, column=0, sticky='w', pady=2)
        self.lbl_power = tk.Label(left_info, text="Laser Power: ", font=lbl_font, bg='#f0f0f0')
        self.lbl_power.grid(row=2, column=0, sticky='w', pady=2)
        self.lbl_features = tk.Label(left_info, text="MPL: --, MPW: --, AVL: --, MXL: --",
                                     font=lbl_font_small, fg="gray", bg='#f0f0f0')
        self.lbl_features.grid(row=3, column=0, sticky='w', pady=5)

        # 右栏：预测结果
        right_info = tk.Frame(info_frame, bg='#f0f0f0')
        right_info.pack(side=tk.RIGHT, anchor=tk.E, padx=10)

        self.lbl_pred_title = tk.Label(right_info, text="LSTM Prediction", font=lbl_font_title, bg='#f0f0f0')
        self.lbl_pred_title.grid(row=0, column=0, sticky='w', pady=(0, 8))
        self.lbl_dilution = tk.Label(right_info, text="Dilution ratio: ", font=lbl_font, fg="blue", bg='#f0f0f0')
        self.lbl_dilution.grid(row=1, column=0, sticky='w', pady=2)
        self.lbl_depth = tk.Label(right_info, text="Depth: ", font=lbl_font, fg="blue", bg='#f0f0f0')
        self.lbl_depth.grid(row=2, column=0, sticky='w', pady=2)
        self.lbl_height = tk.Label(right_info, text="Height: ", font=lbl_font, fg="blue", bg='#f0f0f0')
        self.lbl_height.grid(row=3, column=0, sticky='w', pady=2)

        # 底部导航按钮
        nav_frame = tk.Frame(self.root, bg='#f0f0f0')
        nav_frame.pack(pady=15)

        self.btn_prev = tk.Button(nav_frame, text="◄ Previous", command=self.prev_image,
                                  font=("Segoe UI", 11), state=tk.DISABLED, bg="#2196F3", fg="white", padx=8)
        self.btn_prev.pack(side=tk.LEFT, padx=10)

        self.btn_next = tk.Button(nav_frame, text="Next ►", command=self.next_image,
                                  font=("Segoe UI", 11), state=tk.DISABLED, bg="#2196F3", fg="white", padx=8)
        self.btn_next.pack(side=tk.LEFT, padx=10)

        self.lbl_progress = tk.Label(nav_frame, text="", bg='#f0f0f0', font=("Segoe UI", 9))
        self.lbl_progress.pack(side=tk.RIGHT, padx=30)

    # ---------- 文件夹选择与批量处理 ----------
    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Select a folder containing images to identify")
        if not folder_path:
            return
        self.img_folder = folder_path
        self.lbl_folder.config(text=folder_path)

        # 重置数据
        self.img_files.clear()
        self.img_abs_paths.clear()
        self.features = None
        self.valid_indices.clear()
        self.predictions = None
        self.pred_mapping.clear()
        self.current_idx = 0

        # 获取所有图片并按文件名排序
        all_files = sorted(os.listdir(folder_path))
        for fname in all_files:
            if fname.lower().endswith(SUPPORTED_IMG_EXT):
                self.img_files.append(fname)
                self.img_abs_paths.append(os.path.join(folder_path, fname))

        if not self.img_files:
            messagebox.showinfo("Information", "No supported image files found in the selected folder.")
            return

        print(f"Found {len(self.img_files)} images, starting batch process...")
        self.batch_process()
        self.display_image(self.current_idx)

    def batch_process(self):
        """
        对所有图片执行：
          1. YOLO 检测 → MPL, MPW
          2. 亮度计算 → AVL, MXL
          3. 解析文件名 → layers, power
        得到 6 维特征矩阵，然后用 LSTM 时序预测
        """
        n = len(self.img_files)
        features_all = np.zeros((n, 6))
        is_valid = np.zeros(n, dtype=bool)

        for i, img_path in enumerate(self.img_abs_paths):
            # 读取图片
            try:
                img_pil = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Cannot open image {img_path}: {e}")
                continue

            # YOLO 检测
            results = self.yolo(img_path, verbose=False)  # 返回列表，单图片长度1
            det = results[0]

            # 提取熔池尺寸
            mpl, mpw = extract_detection_features(det)
            if mpl is None:
                print(f"Warning: {self.img_files[i]} no target detected, will be skipped.")
                continue

            # 亮度
            avg_lum, max_lum = calculate_brightness(img_pil)

            # 解析文件名
            try:
                layers, power = parse_filename(self.img_files[i])
            except ValueError as e:
                print(f"Filename parse error: {self.img_files[i]} - {e}")
                continue

            # 特征顺序: [MPL, MPW, AVL, MXL, Laser Power, Number of layers]
            feat = np.array([mpl, mpw, avg_lum, max_lum, power, layers], dtype=np.float32)
            features_all[i] = feat
            is_valid[i] = True

        valid_idx = np.where(is_valid)[0]
        if len(valid_idx) == 0:
            messagebox.showerror("Error", "No image could be successfully processed. Check your images and models.")
            return

        valid_features = features_all[valid_idx]

        # 检查样本量是否满足 look_back
        if len(valid_features) < self.look_back:
            messagebox.showwarning("Insufficient Samples",
                                   f"At least {self.look_back} valid images are required for LSTM prediction, "
                                   f"but only {len(valid_features)} are available.")
            # 即使不能预测，仍可显示图片

        try:
            pred_array = self.lstm_predict(valid_features)
        except Exception as e:
            messagebox.showerror("Prediction Failed", str(e))
            return

        # 存储结果并建立映射
        self.features = features_all
        self.valid_indices = valid_idx.tolist()
        self.predictions = pred_array

        # 预测结果对应关系：前 look_back 个有效样本无预测值
        pred_offset = self.look_back
        self.pred_mapping.clear()
        for i, idx in enumerate(self.valid_indices):
            if i >= self.look_back:
                self.pred_mapping[idx] = pred_array[i - pred_offset]
            else:
                self.pred_mapping[idx] = None

        # 更新界面状态
        self.update_nav_buttons()

        msg = (f"Processed! Total {n} images, {len(valid_idx)} valid, "
               f"{max(0, len(valid_idx)-self.look_back)} have predictions.")
        print(msg)
        self.lbl_progress.config(text=msg)

    def lstm_predict(self, features_array):
        """
        使用 LSTM 模型进行滑动窗口预测。
        参数:
            features_array: shape (n_samples, 6)
        返回:
            pred_array: shape (n_samples - look_back, 3)
        """
        n = features_array.shape[0]
        if n < self.look_back:
            raise ValueError(f"Sample size {n} < look_back {self.look_back}")

        scaled = self.feature_scaler.transform(features_array)
        X = []
        for i in range(self.look_back, n):
            X.append(scaled[i - self.look_back:i, :])
        X = np.array(X)

        y_scaled = self.lstm_model.predict(X, verbose=0)
        y_pred = self.target_scaler.inverse_transform(y_scaled)
        return y_pred

    # ---------- 图片显示与导航 ----------
    def display_image(self, idx):
        """在主窗口显示第 idx 张图片及其检测/预测信息"""
        if idx < 0 or idx >= len(self.img_files):
            return

        self.current_idx = idx
        img_path = self.img_abs_paths[idx]
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except Exception:
            messagebox.showerror("Error", f"Cannot open image: {img_path}")
            return

        # 重新检测以绘制边框（或可缓存结果，为简单起见重新运行）
        try:
            results = self.yolo(img_path, verbose=False)
            img_with_boxes = draw_boxes(img_pil.copy(), results[0])
        except Exception:
            img_with_boxes = img_pil.copy()

        # 缩放图片
        display_width = IMAGE_DISPLAY_WIDTH
        w, h = img_with_boxes.size
        ratio = display_width / w
        new_h = int(h * ratio)
        img_resized = img_with_boxes.resize((display_width, new_h), Image.Resampling.LANCZOS)

        # 更新 Canvas
        self.photo_ref = ImageTk.PhotoImage(img_resized)
        self.canvas.config(width=display_width, height=new_h)
        self.canvas.delete("all")
        self.canvas.create_image(display_width // 2, new_h // 2, image=self.photo_ref, anchor=tk.CENTER)

        # 更新左侧信息
        self.lbl_filename.config(text=f"Image: {self.img_files[idx]}")
        try:
            layers, power = parse_filename(self.img_files[idx])
            self.lbl_layers.config(text=f"Number of layers: {layers}")
            self.lbl_power.config(text=f"Laser Power: {power}")
        except ValueError:
            self.lbl_layers.config(text="Number of layers: ?")
            self.lbl_power.config(text="Laser Power: ?")

        if self.features is not None and idx in self.valid_indices:
            feat = self.features[idx]
            self.lbl_features.config(text=f"MPL: {feat[0]:.1f}, MPW: {feat[1]:.1f}, "
                                           f"AVL: {feat[2]:.1f}, MXL: {feat[3]:.1f}")
        else:
            self.lbl_features.config(text="MPL: --, MPW: --, AVL: --, MXL: --")

        # 更新右侧预测结果
        if self.pred_mapping and idx in self.pred_mapping and self.pred_mapping[idx] is not None:
            pred = self.pred_mapping[idx]
            self.lbl_dilution.config(text=f"Dilution ratio: {pred[0]:.4f}")
            self.lbl_depth.config(text=f"Depth: {pred[1]:.4f}")
            self.lbl_height.config(text=f"Height: {pred[2]:.4f}")
        else:
            self.lbl_dilution.config(text="Dilution ratio: --")
            self.lbl_depth.config(text="Depth: --")
            self.lbl_height.config(text=f"Height: -- (need {self.look_back} prior valid frames)")

        self.update_nav_buttons()

    def next_image(self):
        if self.current_idx < len(self.img_files) - 1:
            self.display_image(self.current_idx + 1)

    def prev_image(self):
        if self.current_idx > 0:
            self.display_image(self.current_idx - 1)

    def update_nav_buttons(self):
        if len(self.img_files) == 0:
            self.btn_prev.config(state=tk.DISABLED)
            self.btn_next.config(state=tk.DISABLED)
            return
        self.btn_prev.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_idx < len(self.img_files) - 1 else tk.DISABLED)
        self.lbl_progress.config(text=f"{self.current_idx + 1} / {len(self.img_files)}")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedApp(root)
    root.mainloop()