import pandas as pd
import numpy as np
import tqdm
from tqdm import auto
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# 日期,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數
# 113/01/02,27997826,16549619798,590.00,593.00,589.00,593.00, 0.00,20667
df = pd.read_csv("stock_day_2330_2024.csv")
print(df.isna().sum())
#
df.drop(["成交股數", "成交金額"], axis=1, inplace=True) # inplace=True 表示不用 asign
df.columns = ["Date", "Open", "High", "Low", "Close", "RD", "Volumn"]
df.set_index(["Date"], inplace=True)
# 将指定列转换为数字类型（例如，将 'column_name' 列转换为数字）
df["Open"] = pd.to_numeric(df["Open"], errors='coerce')
# 如果要将多个列转换为数字类型，可以使用以下方法：
df[["Open", "High", "Low", "Close", "RD"]] = df[["Open", "High", "Low", "Close", "RD"]].apply(pd.to_numeric, errors='coerce')

# 检查 DataFrame 的数据类型
print(df.dtypes)
#應變量
df.RD.plot()

#先拆分，再正規化
train = df[0:100]
test = df[100:]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train = pd.DataFrame(scaler.fit_transform(train), columns=df.columns)
test = pd.DataFrame(scaler.fit_transform(test), columns=df.columns)

#製造X跟Y(RD)
n = 10 #改n即可，資料1/4起，所以能預測的第一個Y為2/3，抓30天
feature_names = list(train.drop('RD', axis=1).columns)
X = []
y = []
indexes = []
norm_data_x = train[feature_names]
for i in auto.tqdm(range(0,len(train)-n)): 
  X.append(norm_data_x.iloc[i:i+n]. values) 
  y.append(train['RD'].iloc[i+n-1]) #現有資料+30天的Y
  indexes.append(train.index[i+n-1]) #Y的日期

print(X[0])
print(y[0])

X=np.array(X) # RD外的6個自變量，記憶體=30，EX:預測12/31的Y，用12/1~12/30的X
y=np.array(y)
print(X.shape)
print(y.shape)

# Adding the LSTM layer
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
n_steps = 10 
n_features = 5
model = Sequential()
model.add(LSTM(50,activation='relu', return_sequences=False, input_shape = (n_steps, n_features)))
#input_shape = (n_steps, n_features)  幾步, 幾個特徵
# return_sequences預設false，輸出是否為序列? 是: 輸出多個值
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse' , metrics=['mse'])

history = model.fit(X,y,batch_size=100,epochs=20)



#model = Sequential()
#model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#model.add(Dropout(0.2))
#model.add(LSTM(50, return_sequences=True))

# 鳶尾花
#model.add(Dense(64, input_dim=3, activation='relu'))

#
#n_steps = 3; n_features = 1
#model.add(LSTM(50, activation='relu',  input_shape = (n_steps, n_features)) )
#
#model.add(Dense(1))
