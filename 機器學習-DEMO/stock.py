import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(10)  # 指定亂數種子
# 載入糖尿病資料集
df = pd.read_csv("./2330_stock_flag.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 1:9]
Y = dataset[:, 9]
# 確保 X 和 Y 都是 numpy 陣列
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)
# 特徵標準化 (CH5_2_3.py)
X -= X.mean(axis=0)
X /= X.std(axis=0)
# 定義模型 (隱藏層=>relu, 輸出層=> sigmoid/tanh二元分類, Softmax 多元分類 )
model = Sequential()
model.add(Dense(10, input_shape=(8,), activation="relu")) # 1st隱藏層/神經元10/特徵8/啟動函數(relu)/全連接層
model.add(Dense(8, activation="relu"))                    # 2nd隱藏層/神經元8/      啟動函數(relu)
model.add(Dense(1, activation="sigmoid"))                 #    輸出層/神經元1/      啟動函數(sigmoid)
model.summary()  # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
# 訓練模型
model.fit(X, Y, epochs=10, batch_size=5)
# 評估模型
loss, accuracy = model.evaluate(X, Y)
print("損失 = {:.2f}".format(loss))
print("準確度 = {:.2f}".format(accuracy))


"""
# 讀取 CSV 檔案
df = pd.read_csv('./2330_stock_new.csv')

# 計算每筆資料的 Flag
def calculate_flag(index):
    if index >= len(df) - 5:
        return 0  # 如果剩餘資料不足五筆，無法進行比較
    current_close = df.iloc[index]['Close']
    # 檢查接下來五筆資料中的 Close 值是否超過 4%
    for i in range(index + 1, index + 6):
        if i < len(df) and df.iloc[i]['Close'] > current_close * 1.04:
            return 1
    return 0

# 應用 calculate_flag 函數到每一筆資料
df['Flag'] = df.index.to_series().apply(calculate_flag)

# 儲存結果為新的 CSV 檔案
df.to_csv('./2330_stock_flag.csv', index=False)
"""
