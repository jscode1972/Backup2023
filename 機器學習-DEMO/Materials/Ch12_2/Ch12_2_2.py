import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

np.random.seed(10)  # 指定亂數種子
# 載入Google股價的訓練資料集
df_train = pd.read_csv("GOOG_Stock_Price_Train.csv",
                       index_col="Date",parse_dates=True)
X_train_set = df_train.iloc[:,4:5].values  # Adj Close欄位
# 特徵標準化 - 正規化
sc = MinMaxScaler() 
X_train_set = sc.fit_transform(X_train_set)
# 取出幾天前股價來建立成特徵和標籤資料集
def create_dataset(ds, look_back=1):
    X_data, Y_data = [],[]
    for i in range(len(ds)-look_back):
        X_data.append(ds[i:(i+look_back), 0])
        Y_data.append(ds[i+look_back, 0])
    
    return np.array(X_data), np.array(Y_data)

look_back = 60
print("回看天數:", look_back)
# 分割成特徵資料和標籤資料
X_train, Y_train = create_dataset(X_train_set, look_back)
# 轉換成(樣本數, 時步, 特徵)張量
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)
# 定義模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, 
               input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="mse", optimizer="adam")
# 訓練模型
model.fit(X_train, Y_train, epochs=100, batch_size=32)
# 使用模型預測股價 - 2017年1~3月預測 4 月份股價
df_test = pd.read_csv("GOOG_Stock_Price_Test.csv")
X_test_set = df_test.iloc[:,4:5].values
# 產生特徵資料和標籤資料
X_test, Y_test = create_dataset(X_test_set, look_back)
X_test = sc.transform(X_test)
# 轉換成(樣本數, 時步, 特徵)張量
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test_pred = model.predict(X_test)
# 將預測值轉換回股價
X_test_pred_price = sc.inverse_transform(X_test_pred)
# 繪出股價圖表
import matplotlib.pyplot as plt

plt.plot(Y_test, color="red", label="Real Stock Price")
plt.plot(X_test_pred_price, color="blue", label="Predicted Stock Price")
plt.title("2017 Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Time Price")
plt.legend()
plt.show()
