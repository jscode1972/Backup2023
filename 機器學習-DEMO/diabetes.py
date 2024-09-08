import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# 官網: https://www.tensorflow.org/tutorials/quickstart/advanced
# 官網: https://keras.io/api/

np.random.seed(10)  # 指定亂數種子
# 載入糖尿病資料集
df = pd.read_csv("./diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:8]
Y = dataset[:, 8]  # 神經元1
# 特徵標準化 (CH5_2_3.py)
X -= X.mean(axis=0)
X /= X.std(axis=0)
# One-hot編碼 sigmoid => softmax
Y = to_categorical(Y) # 搭配輸出 神經元2 (CH5_2_3a.py)
# 分割訓練和測試資料集
x_train, y_train = X[:690], Y[:690] # 訓練資料 前 690 筆
x_test,  y_test = X[690:], Y[690:]  # 測試資料 後 78 筆
# 定義模型 (隱藏層=>relu, 輸出層=> sigmoid/tanh二元分類, Softmax 多元分類 )
model = Sequential()
model.add(Dense(8,
                #8,
                input_shape=(8,), 
                #kernel_initializer="random_uniform", # 預設: glorot_uniform (CH5_2_3b.py 權重初始器）) 差異不大 https://keras.io/initializers
                #bias_initializer="ones",             # 預設: zeros          (CH5_2_3b.py 權重初始器）) 差異不大 https://keras.io/initializers
                activation="relu"))                   # 1st隱藏層/神經元10/特徵8/啟動函數(relu)/全連接層
model.add(Dense(#8,                                    # 神經元8 => 樣本數少不需要那麼多 
                6,                                   # 神經元6 => 減少神經網路參數量 
                #kernel_initializer="random_uniform", # 預設: glorot_uniform (CH5_2_3b.py 權重初始器）) 差異不大 https://keras.io/initializers
                #bias_initializer="ones",             # 預設: zeros          (CH5_2_3b.py 權重初始器）) 差異不大 https://keras.io/initializers
                activation="relu"))                   # 2nd隱藏層/神經元8/    啟動函數(relu)                 
model.add(Dense(#1,                                   # 神經元1
                2,                                    # 神經元2
                #kernel_initializer="random_uniform", # 預設: glorot_uniform (CH5_2_3b.py 權重初始器）) 差異不大 https://keras.io/initializers
                #bias_initializer="ones",             # 預設: zeros          (CH5_2_3b.py 權重初始器）) 差異不大 https://keras.io/initializers
                #activation="sigmoid"                 # 輸出層/神經元1/       啟動函數(sigmoid)   
                activation="softmax"))                # 輸出層/神經元2/       (CH5_2_3a.py 啟動函數)     差異不大  softmax 
model.summary()  # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="binary_crossentropy", 
              #optimizer="sgd",                       # 優化器 https://keras.io/initializers (sgd/adam/rmsprop)
              optimizer="adam",                       # 優化器 https://keras.io/initializers (sgd/adam/rmsprop)
              metrics=["accuracy"])
# 1.1 訓練模型 (全批訓練)
#model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
# 1.2 評估模型 (全批訓練)
#loss, accuracy = model.evaluate(X, Y)
#print("全批訓練-損失 = {:.2f}".format(loss))
#print("全批訓練-準確度 = {:.2f}".format(accuracy))

# 2.1 訓練模型 (訓練+測試) CH5_2_4.py
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),    # 過度擬合 ＝> 改這個 => 好像沒改善
                    epochs=10, batch_size=10, verbose=1) # 過度擬合 150 => 10
# 2.2 評估模型 (訓練前半部)
loss, accuracy = model.evaluate(x_train, y_train)
print("前半訓練-損失 = {:.2f}".format(loss))
print("前半訓練-準確度 = {:.2f}".format(accuracy))   # 0.85
# 評估模型 (測試吼半部)
loss, accuracy = model.evaluate(x_test, y_test)
print("後半測試-損失 = {:.2f}".format(loss))
print("後半測試-準確度 = {:.2f}".format(accuracy))  # 0.73 (筆訓練低 => 過度擬合)

# 顯示訓練和驗證損失的圖表
import matplotlib.pyplot as plt

loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# 顯示訓練和驗證準確度
acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "b-", label="Training Acc")
plt.plot(epochs, val_acc, "r--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 問題種類        輸出層啟動函數   損失函數
# ------------  --------------  ---------------
# 二元分類        sigmoid        binary_crossentropy
# 單標籤多元分類   softmax        categorical_crossentropy
# 多標籤多元分類   sigmoid        binary_crossentropy
# 回歸分析        不需要          mse
# 回歸值在 0..1   sigmoid        mse 或 binary_crossentropy
