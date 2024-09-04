import pandas as pd
import numpy as np
df = pd.read_csv("2317.csv")
df.isna().sum()
#%%
df.drop(['Change'], axis=1, inplace=True) #上期比
df.columns = ['date','Open','High','Low','Close','RD','Volume']
df
#%%
# 將日期設為index
df.set_index(['date'], inplace=True)
df
#%%
#應變量
df.RD.plot()
#%%
#先拆分，再正規化
train = df[0:2670]
test = df[2670:]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train = pd.DataFrame(scaler.fit_transform(train), columns=df.columns)
test = pd.DataFrame(scaler.fit_transform(test), columns=df.columns)
#%%
#製造X跟Y(RD)
import tqdm
n = 30 #改n即可，資料1/4起，所以能預測的第一個Y為2/23，抓30天
feature_names = list(train.drop('RD', axis=1).columns)
X = []
y = []
indexes = []
norm_data_x = train[feature_names]
for i in tqdm.tqdm_notebook(range(0,len(train)-n)): 
  X.append(norm_data_x.iloc[i:i+n]. values) 
  y.append(train['RD'].iloc[i+n]) #現有資料+30天的Y
  indexes.append(train.index[i+n]) #Y的日期

print(X[0])
print(y[0])

X=np.array(X) # RD外的6個自變量，記憶體=30，EX:預測12/31的Y，用12/1~12/30的X
y=np.array(y)
print(X.shape)
print(y.shape)
#%%
# Adding the LSTM layer
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
n_steps = 30 
n_features = 5
model = Sequential()
model.add(LSTM(50,activation='relu', return_sequences=False, input_shape = (n_steps, n_features)))
#input_shape = (n_steps, n_features)  幾步, 幾個特徵
# return_sequences預設false，輸出是否為序列? 是: 輸出多個值
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse' , metrics=['mse','mape'])
#%%
history = model.fit(X,y,batch_size=100,epochs=20)

# 顯示loss
import matplotlib.pyplot as plt

plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.plot(history.history["loss"])
#%%
#製造X跟Y(RD)
import tqdm
n = 30 #改n即可，資料1/4起，所以能預測的第一個Y為2/3，抓30天
feature_names = list(test.drop('RD', axis=1).columns)
X = []
y = []
indexes = []
norm_data_x = test[feature_names]
for i in tqdm.tqdm_notebook(range(0,len(test)-n)): 
  X.append(norm_data_x.iloc[i:i+n].values) 
  y.append(test['RD'].iloc[i+n-1]) #現有資料+30天的Y
  indexes.append(test.index[i+n-1]) #Y的日期
X=np.array(X) # RD外的6個自變量，記憶體=30，EX:預測12/31的Y，用12/1~12/30的X
y=np.array(y)

predictions = model.predict(X)
predictions = pd.DataFrame(predictions).rename(columns={0: '預測值'})
Y_test = pd.DataFrame(y).rename(columns={0: '實際值'})

final = pd.concat([predictions,Y_test],axis=1)
final['mae'] = abs(final['預測值'] - final['實際值'])
final
#norm_data = pd.DataFrame(scaler.inverse_transform(norm_data), columns=df.columns, index=df.index)
