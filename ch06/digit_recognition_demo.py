# coding: utf-8
"""
手書き数字認識デモ

使い方:
1. 先に train_and_save_model.py を実行してモデルを訓練・保存してください
2. このスクリプトを実行すると、MNISTテストデータからランダムに選んだ画像で予測を行います
3. tkinterが利用可能な環境では digit_gui.py を実行してください（手描き入力が可能）

Requirements:
- numpy
- PIL (Pillow)
- matplotlib (画像表示用、オプション)
"""

import os
import sys
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from digit_classifier import DigitClassifier
from dataset.mnist import load_mnist

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available - images will not be displayed")

class DigitRecognitionDemo:
    def __init__(self):
        print("手書き数字認識デモを初期化中...")
        self.classifier = DigitClassifier()
        
        # MNISTデータの読み込み
        print("MNISTデータを読み込み中...")
        (_, _), (self.x_test, self.t_test) = load_mnist(normalize=False)
        print(f"テストデータ: {len(self.x_test)}枚の画像")
    
    def predict_random_images(self, num_images=10):
        """ランダムに選択した画像で予測を行う"""
        print(f"\n{num_images}枚のランダムな画像で予測を実行...")
        
        correct_count = 0
        indices = random.sample(range(len(self.x_test)), num_images)
        
        for i, idx in enumerate(indices, 1):
            image = self.x_test[idx].reshape(28, 28)
            true_label = self.t_test[idx]
            
            result = self.classifier.predict(image)
            predicted_digit = result['predicted_digit']
            confidence = result['probabilities'][predicted_digit] * 100
            
            is_correct = predicted_digit == true_label
            if is_correct:
                correct_count += 1
            
            print(f"\n画像 {i}:")
            print(f"  正解: {true_label}")
            print(f"  予測: {predicted_digit} (信頼度: {confidence:.1f}%)")
            print(f"  結果: {'✓ 正解' if is_correct else '✗ 不正解'}")
            
            # 上位3位の確率を表示
            top_indices = np.argsort(result['probabilities'])[::-1][:3]
            print(f"  トップ3予測:")
            for rank, digit_idx in enumerate(top_indices, 1):
                prob = result['probabilities'][digit_idx] * 100
                print(f"    {rank}位: {digit_idx} ({prob:.1f}%)")
        
        accuracy = (correct_count / num_images) * 100
        print(f"\n精度: {correct_count}/{num_images} = {accuracy:.1f}%")
        
        return accuracy
    
    def interactive_demo(self):
        """対話的デモ"""
        print("\n対話的デモモード")
        print("コマンド:")
        print("  'r' または 'random': ランダムな画像で予測")
        print("  'q' または 'quit': 終了")
        print("  数字(0-9999): 指定したインデックスの画像で予測")
        
        while True:
            try:
                user_input = input("\nコマンドを入力してください: ").strip().lower()
                
                if user_input in ['q', 'quit']:
                    print("デモを終了します。")
                    break
                
                elif user_input in ['r', 'random']:
                    idx = random.randint(0, len(self.x_test) - 1)
                    self.predict_single_image(idx)
                
                elif user_input.isdigit():
                    idx = int(user_input)
                    if 0 <= idx < len(self.x_test):
                        self.predict_single_image(idx)
                    else:
                        print(f"インデックスは0-{len(self.x_test)-1}の範囲で入力してください。")
                
                else:
                    print("無効なコマンドです。")
                    
            except KeyboardInterrupt:
                print("\nデモを終了します。")
                break
            except Exception as e:
                print(f"エラー: {e}")
    
    def predict_single_image(self, idx):
        """指定したインデックスの画像で予測"""
        image = self.x_test[idx].reshape(28, 28)
        true_label = self.t_test[idx]
        
        result = self.classifier.predict(image)
        predicted_digit = result['predicted_digit']
        
        print(f"\n画像インデックス: {idx}")
        print(f"正解ラベル: {true_label}")
        print(f"予測結果: {predicted_digit}")
        print(f"判定: {'正解' if predicted_digit == true_label else '不正解'}")
        
        print("各数字の確率:")
        for digit, prob in enumerate(result['probabilities']):
            marker = " ← 予測" if digit == predicted_digit else ""
            marker += " (正解)" if digit == true_label else ""
            print(f"  {digit}: {prob*100:5.1f}%{marker}")
        
        # matplotlib が利用可能な場合は画像を表示
        if HAS_MATPLOTLIB:
            try:
                plt.figure(figsize=(4, 4))
                plt.imshow(image, cmap='gray')
                plt.title(f"Index: {idx}, True: {true_label}, Predicted: {predicted_digit}")
                plt.axis('off')
                plt.show()
            except:
                print("画像の表示に失敗しました。")

def main():
    """メイン関数"""
    print("="*50)
    print("手書き数字認識デモ")
    print("="*50)
    
    try:
        demo = DigitRecognitionDemo()
        
        # 最初に少数の画像でテスト
        print("\n=== 初期テスト（5枚の画像）===")
        demo.predict_random_images(5)
        
        # 対話的デモ
        print("\n=== 対話的デモ ===")
        demo.interactive_demo()
        
    except FileNotFoundError as e:
        print("エラー: 学習済みモデルが見つかりません。")
        print("先に train_and_save_model.py を実行してモデルを訓練してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()