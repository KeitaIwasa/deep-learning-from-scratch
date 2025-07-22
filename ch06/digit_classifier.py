# coding: utf-8
import os
import sys
import pickle
import numpy as np
from PIL import Image

sys.path.append(os.pardir)
from common.multi_layer_net import MultiLayerNet

class DigitClassifier:
    """学習済みモデルを使用した数字認識クラス"""
    
    def __init__(self, model_path='trained_model.pkl'):
        """
        Parameters
        ----------
        model_path : str
            学習済みモデルファイルのパス
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """学習済みモデルを読み込む"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            print(f"Model file {self.model_path} not found. Please train the model first.")
            sys.exit(1)
    
    def preprocess_image(self, image_data, size=(28, 28)):
        """
        画像データを前処理してモデル入力用に変換
        
        Parameters
        ----------
        image_data : numpy.ndarray
            2D画像データ (0-255の値)
        size : tuple
            リサイズ後のサイズ (高さ, 幅)
            
        Returns
        -------
        numpy.ndarray
            前処理済み画像データ (784次元のベクトル、0-1に正規化)
        """
        # PIL Imageに変換してリサイズ
        if image_data.max() > 1:
            image_data = image_data.astype(np.uint8)
        
        img = Image.fromarray(image_data)
        img = img.resize(size, Image.LANCZOS)
        
        # numpy配列に戻して正規化
        processed_img = np.array(img, dtype=np.float32)
        processed_img = processed_img / 255.0
        
        # 784次元のベクトルに変換
        return processed_img.reshape(1, -1)
    
    def predict(self, image_data):
        """
        画像データから数字を予測
        
        Parameters
        ----------
        image_data : numpy.ndarray
            2D画像データ
            
        Returns
        -------
        dict
            予測結果 {'predicted_digit': int, 'probabilities': numpy.ndarray}
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load model first.")
        
        # 前処理
        processed_img = self.preprocess_image(image_data)
        
        # 予測
        output = self.model.predict(processed_img)
        probabilities = self.softmax(output[0])
        predicted_digit = np.argmax(probabilities)
        
        return {
            'predicted_digit': int(predicted_digit),
            'probabilities': probabilities
        }
    
    def softmax(self, x):
        """ソフトマックス関数"""
        exp_x = np.exp(x - np.max(x))  # オーバーフロー対策
        return exp_x / np.sum(exp_x)
    
    def get_confidence_scores(self, image_data):
        """
        各数字の信頼度スコアを取得
        
        Parameters
        ----------
        image_data : numpy.ndarray
            2D画像データ
            
        Returns
        -------
        list
            各数字(0-9)の信頼度スコアのリスト
        """
        result = self.predict(image_data)
        return [(i, float(prob)) for i, prob in enumerate(result['probabilities'])]

if __name__ == "__main__":
    # テスト用コード
    classifier = DigitClassifier()
    print("Digit classifier initialized successfully!")