# coding: utf-8
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from digit_classifier import DigitClassifier

class DigitRecognitionGUI:
    """手書き数字認識GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("手書き数字認識")
        self.root.geometry("600x500")
        
        # 描画用の変数
        self.canvas_size = 280  # 28x28を10倍にスケール
        self.brush_size = 15
        self.drawing = False
        
        # PIL Imageで描画データを管理
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)  # 白背景
        self.draw = ImageDraw.Draw(self.image)
        
        # 分類器の初期化
        try:
            self.classifier = DigitClassifier()
        except SystemExit:
            messagebox.showerror("エラー", "学習済みモデルが見つかりません。先にoverfit_weight_decay.pyを実行してモデルを訓練してください。")
            root.quit()
            return
        
        self.setup_ui()
    
    def setup_ui(self):
        """UIコンポーネントの設定"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # タイトル
        title_label = ttk.Label(main_frame, text="手書き数字認識", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # 左側: 描画エリア
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, padx=(0, 20), sticky=(tk.W, tk.N))
        
        canvas_label = ttk.Label(left_frame, text="数字を描いてください:", font=("Arial", 12))
        canvas_label.grid(row=0, column=0, pady=(0, 5))
        
        # 描画キャンバス
        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            bd=2,
            relief=tk.RAISED
        )
        self.canvas.grid(row=1, column=0, pady=(0, 10))
        
        # マウスイベントのバインド
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # ボタンフレーム
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=2, column=0, pady=5)
        
        # クリアボタン
        clear_btn = ttk.Button(button_frame, text="クリア", command=self.clear_canvas)
        clear_btn.grid(row=0, column=0, padx=(0, 10))
        
        # 予測ボタン
        predict_btn = ttk.Button(button_frame, text="予測", command=self.predict_digit)
        predict_btn.grid(row=0, column=1)
        
        # 右側: 結果表示エリア
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.N, tk.E))
        
        result_label = ttk.Label(right_frame, text="予測結果:", font=("Arial", 12))
        result_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # 予測結果表示
        self.prediction_label = ttk.Label(
            right_frame, 
            text="まだ予測されていません", 
            font=("Arial", 24, "bold"),
            foreground="blue"
        )
        self.prediction_label.grid(row=1, column=0, pady=(0, 20))
        
        # 信頼度表示
        confidence_label = ttk.Label(right_frame, text="各数字の確率:", font=("Arial", 12))
        confidence_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        # 確率表示用のフレーム
        self.confidence_frame = ttk.Frame(right_frame)
        self.confidence_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # 確率表示用ラベルの初期化
        self.confidence_labels = []
        for i in range(10):
            label = ttk.Label(self.confidence_frame, text=f"{i}: 0.00%")
            label.grid(row=i//5, column=i%5, padx=5, pady=2, sticky=tk.W)
            self.confidence_labels.append(label)
    
    def start_draw(self, event):
        """描画開始"""
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
    
    def draw_on_canvas(self, event):
        """キャンバス上に描画"""
        if self.drawing:
            # Tkinterキャンバスに描画
            self.canvas.create_oval(
                event.x - self.brush_size//2, event.y - self.brush_size//2,
                event.x + self.brush_size//2, event.y + self.brush_size//2,
                fill='black', outline='black'
            )
            
            # PIL Imageにも描画（予測用）
            self.draw.ellipse([
                event.x - self.brush_size//2, event.y - self.brush_size//2,
                event.x + self.brush_size//2, event.y + self.brush_size//2
            ], fill=0)  # 黒
            
            self.last_x, self.last_y = event.x, event.y
    
    def stop_draw(self, event):
        """描画終了"""
        self.drawing = False
    
    def clear_canvas(self):
        """キャンバスをクリア"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="まだ予測されていません")
        
        # 確率表示をリセット
        for i, label in enumerate(self.confidence_labels):
            label.config(text=f"{i}: 0.00%")
    
    def predict_digit(self):
        """数字を予測"""
        try:
            # PIL Imageを28x28にリサイズして予測用データに変換
            resized_img = self.image.resize((28, 28), Image.LANCZOS)
            img_array = np.array(resized_img)
            
            # 背景が白(255)、文字が黒(0)なので、MNISTと合わせるために反転
            img_array = 255 - img_array
            
            # 予測実行
            result = self.classifier.predict(img_array)
            predicted_digit = result['predicted_digit']
            probabilities = result['probabilities']
            
            # 結果表示
            self.prediction_label.config(text=f"予測: {predicted_digit}")
            
            # 各数字の確率表示
            for i, prob in enumerate(probabilities):
                color = "red" if i == predicted_digit else "black"
                self.confidence_labels[i].config(
                    text=f"{i}: {prob*100:.1f}%",
                    foreground=color
                )
                
        except Exception as e:
            messagebox.showerror("エラー", f"予測中にエラーが発生しました: {str(e)}")

def main():
    """メイン関数"""
    root = tk.Tk()
    app = DigitRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()