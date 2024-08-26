import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt

class GoogleStockPricePredictor:
    def __init__(self, train_csv, test_csv, look_back=60, seed=10):
        # 初始化模型，設置訓練和測試資料路徑、look_back天數，以及隨機種子
        np.random.seed(seed)
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.look_back = look_back
        self.sc = MinMaxScaler()
        
        # 準備訓練資料
        self.X_train, self.Y_train = self._prepare_training_data()
        # 建立模型
        self.model = self._build_model()
    
    def _prepare_training_data(self):
        # 載入訓練資料集並標準化
        df_train = pd.read_csv(self.train_csv, index_col="Date", parse_dates=True)
        X_train_set = df_train.iloc[:, 4:5].values  # 只取 "Adj Close" 欄位
        X_train_set = self.sc.fit_transform(X_train_set)
        # 創建特徵和標籤資料集
        return self._create_dataset(X_train_set)
    
    def _create_dataset(self, dataset):
        # 根據look_back天數創建時間序列資料集
        X_data, Y_data = [], []
        for i in range(len(dataset) - self.look_back):
            X_data.append(dataset[i:(i + self.look_back), 0])
            Y_data.append(dataset[i + self.look_back, 0])
        return np.array(X_data), np.array(Y_data)
    
    def _build_model(self):
        # 建立LSTM神經網絡模型並添加Dropout層以防止過擬合
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        # 顯示模型摘要資訊
        model.summary()
        return model
    
    def train_model(self, epochs=100, batch_size=32):
        # 將訓練資料轉換為LSTM所需的3D張量形狀，並進行模型訓練
        X_train_reshaped = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        self.model.fit(X_train_reshaped, self.Y_train, epochs=epochs, batch_size=batch_size)
    
    def predict(self):
        # 載入測試資料集，並創建特徵和標籤資料集
        df_test = pd.read_csv(self.test_csv)
        X_test_set = df_test.iloc[:, 4:5].values
        X_test, Y_test = self._create_dataset(X_test_set)
        # 將測試資料標準化
        X_test_scaled = self.sc.transform(X_test)
        # 將測試資料轉換為LSTM所需的3D張量形狀
        X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        # 使用訓練好的模型進行預測
        X_test_pred = self.model.predict(X_test_reshaped)
        # 將預測結果反標準化為股價
        return self.sc.inverse_transform(X_test_pred), Y_test
    
    def plot_results(self, predictions, actual):
        # 繪製實際股價和預測股價的對比圖
        plt.plot(actual, color="red", label="Real Stock Price")
        plt.plot(predictions, color="blue", label="Predicted Stock Price")
        plt.title("2017 Google Stock Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Google Stock Price")
        plt.legend()
        plt.show()

# 使用方法
def evaluate_google_stock_price():
    # 創建預測器實例，並配置相關參數
    predictor = GoogleStockPricePredictor(train_csv="GOOG_Stock_Price_Train.csv", 
                                          test_csv="GOOG_Stock_Price_Test.csv", 
                                          look_back=60)
    # 訓練模型
    predictor.train_model(epochs=100, batch_size=32)
    # 進行股價預測
    predictions, actual = predictor.predict()
    # 繪製結果
    predictor.plot_results(predictions, actual)

if __name__ == "__main__":
    evaluate_google_stock_price()
