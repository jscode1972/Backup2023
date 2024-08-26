"""
Created on Tue Aug 27 00:43:16 2024
@author: wphuang
---- 優點 --------------------------------------------------------------------------------------------------------
代碼結構清晰：  代碼分為數據準備、模型構建、編譯訓練、評估和可視化五個部分，這樣的結構有助於維護和擴展。
私有方法：      _prepare_data 和 _normalize 方法標識為私有，這符合良好的編程習慣，使得它們在類外部不會被意外調用。
靈活的訓練配置： 在 compile_and_train 方法中，提供了 epochs 和 batch_size 這些參數，使得模型訓練的配置更加靈活。
結果可視化：    提供了訓練損失和準確度的可視化函數 plot_loss 和 plot_accuracy，對於理解模型的訓練過程非常有幫助。
----- 建議 --------------------------------------------------------------------------------------------------------
建議:          數據檢查：在 _prepare_data 方法中，你可以考慮加入數據完整性和正確性的檢查，比如檢查是否存在缺失值或無效數據。
異常處理：      可以在數據加載和模型訓練的步驟中添加一些異常處理機制，以應對可能出現的文件讀取錯誤或模型訓練過程中的異常情況。
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
# 顯示圖表來分析模型的訓練過程
import matplotlib.pyplot as plt

class TitanicModelEvaluator:
    def __init__(self, seed, train_csv, test_csv):
        # 載入Titanic的訓練和測試資料集
        np.random.seed(seed)
        self.df_train = pd.read_csv(train_csv) # "./titanic_train.csv")
        self.df_test = pd.read_csv(test_csv)   # "./titanic_test.csv")
        # 分割成特徵資料和標籤資料
        self.X_train, self.Y_train, self.X_test, self.Y_test = self._prepare_data()
        self.model = self._build_model()
    def _prepare_data(self):
        # Prepare and normalize data
        # 如果資料中確實混合了其他型態的資料，你可以強制將它們轉換為數值型：不然會 std => error?
        X_train = self.df_train.iloc[:, :-1].values.astype(float)
        Y_train = self.df_train.iloc[:, -1].values.astype(float)
        X_test = self.df_test.iloc[:, :-1].values.astype(float)
        Y_test = self.df_test.iloc[:, -1].values.astype(float)
        # Normalize data
        X_train = self._normalize(X_train)
        X_test = self._normalize(X_test)
        return X_train, Y_train, X_test, Y_test
    def _normalize(self, data):
        data -= data.mean(axis=0)
        data /= data.std(axis=0)
        return data
    def _build_model(self):
        # Define the model
        model = Sequential([
            Dense(11, input_dim=self.X_train.shape[1], activation="relu"),
            Dense(11, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        model.summary() # 顯示模型摘要資訊
        return model
    def compile_and_train(self, epochs=100, batch_size=10, validation_split=0.2):
        # Compile the model
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        # Train the model
        print("Training...")
        self.history = self.model.fit(self.X_train, self.Y_train, 
                                      validation_split=validation_split, 
                                      epochs=epochs, batch_size=batch_size, 
                                      verbose=0)
    def evaluate(self):
        # Evaluate the model on training and test data
        print("\nEvaluating...")
        train_loss, train_acc = self.model.evaluate(self.X_train, self.Y_train)
        print(f"訓練資料集的準確度 (Training Accuracy) = {train_acc:.2f}")
        test_loss, test_acc = self.model.evaluate(self.X_test, self.Y_test)
        print(f"測試資料集的準確度 (Testing Accuracy) = {test_acc:.2f}")
    def plot_loss(self):
        # Plot training and validation loss 顯示訓練和驗證損失
        plt.plot(self.history.history["loss"], "b-", label="Training Loss")
        plt.plot(self.history.history["val_loss"], "r--", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    def plot_accuracy(self):
        # Plot training and validation accuracy
        plt.plot(self.history.history["accuracy"], "b-", label="Training Accuracy")
        plt.plot(self.history.history["val_accuracy"], "r--", label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

# Usage
def evaluate_model():
    titanic_model = TitanicModelEvaluator(seed=7, train_csv="./titanic_train.csv", test_csv="./titanic_test.csv")
    titanic_model.compile_and_train()
    titanic_model.evaluate()
    titanic_model.plot_loss()
    titanic_model.plot_accuracy()

# 使用範例
if __name__ == "__main__":
    evaluate_model()
