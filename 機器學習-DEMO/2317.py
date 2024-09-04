"""
https://www.youtube.com/watch?v=EDHpfSSD6ZI&list=PL68v9oqhwEhg8ynlgz-3S6_pPKYhIRIg3&index=13
https://www.youtube.com/watch?v=EDHpfSSD6ZI&list=PL68v9oqhwEhg8ynlgz-3S6_pPKYhIRIg3&index=14 
重點整理
- 預測股票價格很難, 改預測震幅 (跟我想想法很接近，設定利潤 4% )
- 清洗資料  日期不當自變量, 但留著參照
- 先拆分，再正規化 (注意! 正規化最好訓練與驗證分開)
- 製造X跟Y  X 每筆包含 30天x5欄 資料) 共2xxx天
- 建立模型  30天(步), 五個特徵 (DNN 稱作 n_dim)
- 若要回傳多個預測, return_sequences=True  (輸出序列)
- 剛開始跑 不需要跑 validation 會跑很慢, 調適參數有限 => 比較有效方式 1.改模型 2.調整特徵 
- 等到真的預測測試資料效果不好 再來檢查 early stoppping + overfitting
- 跑測試資料 2703-2670=33-30天=3 => 有三天可以預測
"""

import pandas as pd
import numpy as np

"""# 2317 鴻海
df = pd.read_csv("2317.csv")
df.isna().sum()
#%% 
df.drop(['Change'], axis=1, inplace=True) #上期比
df.columns = ['date','Open','High','Low','Close','RD','Volume']
df # Ben: 2703 rows × 7 columns
"""

# 台積電 
# 日期,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數
df = pd.read_csv("2330.csv") # 要去掉空格
df.isna().sum()
#%% 
#df.drop(['成交股數', '成交金額'], axis=1, inplace=True) 
df.columns = ['date','Deal','Amount','Open','High','Low','Close','RD','Volume']
df # Ben: 3601 rows × 9 columns

#%%
# 將日期設為index
df.set_index(['date'], inplace=True) # Ben: 日期不當自變量, 但留著參照, index
df # Ben: 2703 rows × 6 columns (設為索引, 就少一欄)

#%%
#應變量
df.RD.plot()

#%%
#先拆分，再正規化
#train = df[0:2670] df.shape[0]-30-3
#test = df[2670:]
cut_idx = df.shape[0]-30-3
train = df[0:cut_idx] 
test = df[cut_idx:]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train = pd.DataFrame(scaler.fit_transform(train), columns=df.columns) # Ben: 已經是分開正規化範例了
test = pd.DataFrame(scaler.fit_transform(test), columns=df.columns)   # Ben: 已經是分開正規化範例了

#%%
#製造X跟Y(RD)
import tqdm
n = 30 # 改n即可，資料1/4起，所以能預測的第一個Y為2/23，抓30天
feature_names = list(train.drop('RD', axis=1).columns)
X = [] # Ben: len(X) => 2,640 (list), len(X[0]) => 30 (ndarray) 即 2,670-30=2,640
y = [] # Ben: len(y) => 2,640 (list)
indexes = []
norm_data_x = train[feature_names]
for i in tqdm.tqdm_notebook(range(0,len(train)-n)): # Ben: 回圈要扣除最後 30 天
  X.append(norm_data_x.iloc[i:i+n].values)          # Ben: 每一回圈就是30筆資料x5欄('Open','High','Low','Close','Volume')
  y.append(train['RD'].iloc[i+n])  # 現有資料+30天的Y (offset, n=30 )
  indexes.append(train.index[i+n]) # Y的日期         (offset, n=30 )

#print(X[0])
#print(y[0])

#%%
# Ben: X: list(2640) * ndarray(30,5) => ndarray(2640, 30, 5)
# Ben: y: list(2640)                 => ndarray(2640,)
X=np.array(X) # RD外的6個自變量，記憶體=30，EX:預測12/31的Y，用12/1~12/30的X 
y=np.array(y) #
print(X.shape)
print(y.shape)

#%%
# Adding the LSTM layer
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
n_steps    = 30  # 30天
n_features = 7   # 五個特徵
model = Sequential()
model.add(LSTM(50,activation='relu', return_sequences=False, input_shape = (n_steps, n_features)))
#input_shape = (n_steps, n_features)  幾步, 幾個特徵
# return_sequences預設false，輸出是否為序列? 是: 輸出多個值 (序列)
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse' , metrics=['mse','mape'])

#%%
history = model.fit(X,y,batch_size=100,epochs=20) # Ben: 開始訓練

# 顯示loss
import matplotlib.pyplot as plt

plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.plot(history.history["loss"])


############## 以下是跑測試資料 #############################
############## 以下是跑測試資料 #############################

#%%
#製造X跟Y(RD) => 跟前面不一樣!!
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

# 跑測試資料 2703-2670=33-30天=3 => 有三天可以預測 (X+y)

predictions = model.predict(X)
predictions = pd.DataFrame(predictions).rename(columns={0: '預測值'})
Y_test = pd.DataFrame(y).rename(columns={0: '實際值'})

final = pd.concat([predictions,Y_test],axis=1)
final['mae'] = abs(final['預測值'] - final['實際值'])
final


# 以上正規化過後預測資料不會秀給人家看, 需要還原
#norm_data = pd.DataFrame(scaler.inverse_transform(norm_data), columns=df.columns, index=df.index)
#xyz = pd.DataFrame(scaler.inverse_transform(test), columns=test.columns, index=test.index)
predictions = model.predict(X)
predictions = pd.DataFrame(predictions).rename(columns={0: 'RD'})
predictions.insert(0, "Deal", None)
predictions.insert(1, "Amount", None)
predictions.insert(2, "Open", None)
predictions.insert(3, "High", None)
predictions.insert(4, "Low", None)
predictions.insert(5, "Close", None)
predictions.insert(7, "Volume", None)			
predictions
xyz = pd.DataFrame(scaler.inverse_transform(predictions), columns=test.columns) # index=test.index)
xyz
#Y_test = pd.DataFrame(y).rename(columns={4: 'RD'})
