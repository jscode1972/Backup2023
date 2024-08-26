import numpy as np
from keras.datasets import reuters
from keras.preprocessing import sequence
from keras.utils import to_categorical

seed = 10
np.random.seed(seed)  # 指定亂數種子
# 載入Reuters資料集
top_words = 10000
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(
                                       num_words=top_words)
# 資料預處理
max_words = 200 
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)
# One-hot編碼
Y_train = to_categorical(Y_train, 46)
Y_test = to_categorical(Y_test, 46)
