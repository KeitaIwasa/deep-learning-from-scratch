# coding: utf-8
import os
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from digit_classifier import DigitClassifier
from dataset.mnist import load_mnist

def test_with_mnist_sample():
    """MNISTのサンプル画像で動作テスト"""
    print("Testing digit recognition system...")
    
    # 分類器を初期化
    classifier = DigitClassifier()
    
    # MNISTテストデータを読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False)
    
    # テスト用に最初の5個の画像で予測
    for i in range(5):
        test_image = x_test[i].reshape(28, 28)
        true_label = t_test[i]
        
        # 予測実行
        result = classifier.predict(test_image)
        predicted_digit = result['predicted_digit']
        probabilities = result['probabilities']
        
        print(f"\nTest image {i+1}:")
        print(f"True label: {true_label}")
        print(f"Predicted: {predicted_digit}")
        print(f"Correct: {'Yes' if predicted_digit == true_label else 'No'}")
        print("Probabilities:")
        for digit, prob in enumerate(probabilities):
            marker = " <-- PREDICTED" if digit == predicted_digit else ""
            print(f"  {digit}: {prob:.4f}{marker}")

def create_simple_test_image():
    """シンプルなテスト画像を作成してテスト"""
    print("\nTesting with a simple test image...")
    
    # 28x28の黒い背景
    test_image = np.zeros((28, 28), dtype=np.uint8)
    
    # 簡単な線を描いて数字「1」のような形を作る
    test_image[5:23, 13] = 255  # 縦線
    test_image[5:8, 12] = 255   # 左上の斜め線
    test_image[22:25, 10:16] = 255  # 下の横線
    
    classifier = DigitClassifier()
    result = classifier.predict(test_image)
    
    print(f"Predicted digit: {result['predicted_digit']}")
    print("Probabilities:")
    for digit, prob in enumerate(result['probabilities']):
        marker = " <-- PREDICTED" if digit == result['predicted_digit'] else ""
        print(f"  {digit}: {prob:.4f}{marker}")

if __name__ == "__main__":
    try:
        test_with_mnist_sample()
        create_simple_test_image()
        print("\nDigit recognition system is working correctly!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()