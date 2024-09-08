from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import timeseries_dataset_from_array

"""
[盤後資訊]
市場成交資訊             https://www.twse.com.tw/zh/trading/historical/fmtqik.html
各類指數日成交量值        https://www.twse.com.tw/zh/trading/historical/bfiamu.html
當日融券賣出與借券賣出成交量值 https://www.twse.com.tw/zh/trading/historical/twtasu.html
台積電 日收盤價及月平均收盤價  https://www.twse.com.tw/zh/trading/historical/stock-day-avg.html
台積電 月成交資訊         https://www.twse.com.tw/zh/trading/historical/fmsrfk.html
個股日本益比、殖利率及股價淨值比                   https://www.twse.com.tw/zh/trading/historical/bwibbu-day.html
台積電 個股日本益比、殖利率及股價淨值比(以個股月查詢) https://www.twse.com.tw/zh/trading/historical/bwibbu.html
每月當日沖銷交易標的及統計  https://www.twse.com.tw/zh/trading/day-trading/twtb4u-month.html
融資融券餘額 信用交易統計   https://www.twse.com.tw/zh/trading/margin/mi-margn.html

[三大法人]
三大法人買賣金額統計表 https://www.twse.com.tw/zh/trading/foreign/bfi82u.html
三大法人買賣超日報     https://www.twse.com.tw/zh/trading/foreign/t86.html
....投信/自營商
外資期權未平倉量 (空單/多單) https://www.youtube.com/watch?v=8X24ty4vyjc
很多

[指數]
電子類指數及金融保險類指數 https://www.twse.com.tw/zh/indices/taiex/eftri-hist.html
發行量加權股價指數歷史資料 https://www.twse.com.tw/zh/indices/taiex/mi-5min-hist.html
未含金融電子股指數歷史資料 https://www.twse.com.tw/zh/indices/taiex/twt91u.html
電子類報酬指數及金融保險類報酬指數 https://www.twse.com.tw/zh/indices/taiex/eftri.html

[其他]
證券編碼查詢     https://www.twse.com.tw/zh/products/code/query.html
本國上市證券國際證券辨識號碼一覽表 https://isin.twse.com.tw/isin/C_public.jsp?strMode=2
基本市況報導網站 https://mis.twse.com.tw/stock/index?lang=zhHant

[LSTM]
每日收益率、3 天 MA、5 天 MA、10 天 MA、25 天 MA、50 天 MA
這篇文章看看 https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

[常見技術指標建議]
移動平均線（MA, EMA）：5/10/20
  移動平均線能反映價格的趨勢。常用的有簡單移動平均線（SMA）和指數移動平均線（EMA）。
  例如，短期與長期的 MA 或 EMA 之間的差異（如 5 天 vs. 20 天）可以提供價格趨勢的信號。
相對強弱指數（RSI）： 14 天
  RSI 是用來判斷市場是否超買或超賣的指標。當 RSI 接近 70 時，市場可能超買；當 RSI 接近 30 時，市場可能超賣。
隨機指數（Stochastic Oscillator）：
  用來判斷收盤價在一定期間內的相對位置，類似 RSI，能反映超買或超賣狀態。
布林帶（Bollinger Bands）：
  布林帶由移動平均線和兩個標準差組成，能夠反映市場的波動情況。價格靠近上下軌時，可能是反轉信號。
移動平均收斂/發散指標（MACD）：
  MACD 反映兩條不同週期的移動平均線之間的差異，常用於判斷趨勢的強弱和方向。
成交量加權平均價（VWAP）：
  該指標將成交量與價格結合，能反映實際的市場平均交易價，常用於高頻交易和短期決策。
平均真實波幅（ATR）：
  ATR 反映了市場波動性，能幫助捕捉市場的異常波動。
  
[改善 loss 函數的建議]
  選擇合適的損失函數也很重要。(Ben => Huber, LogCosh 有效)
    如果你目前使用的是均方誤差（MSE），你可以考慮其他函數（如 Huber 損失或 LogCosh 損失），以減少異常值對模型的影響。
  總結來說，Early Stopping、正則化、減少複雜度、調整學習率，以及數據增強是常見的改善策略
  Early Stopping 
    由於模型在第 1 個 epoch 時驗證損失最低，並在後面逐漸變差，可以考慮使用**早停法（Early Stopping）**來防止模型過擬合。
  增加正則化 (Ben: L2 => 有效)
    你已經使用了 Dropout，這有助於防止過擬合。如果過擬合仍然嚴重，可以考慮進一步增加 Dropout 比例，或者加入L2 正則化來限制模型的權重值。
  減少模型複雜度 (32->16)
    你的模型可能過於複雜，尤其是在輸入數據較少的情況下。
    你可以嘗試減少 LSTM 單元數量或調整模型的層數，使其更加簡單。對於一些時間序列任務，過多的 LSTM 單元會增加模型的學習能力，但也會更容易過擬合。
[其他評估指標]
  準確率（Accuracy） 和 混淆矩陣 可以幫助你評估模型預測的方向性是否正確（即漲跌預測的準確性）。
  AUC-ROC 曲線 或 F1 Score 等指標有助於衡量模型在二分類問題（如漲或跌）上的表現。  
  均方根誤差（RMSE） 是另一個常用的衡量預測誤差的指標，它可以幫助理解模型在預測具體值方面的表現。
[最佳組合] val_loss: 2.9074 => 1.6190
  step = 1    # 每筆 1 天
  past = 20  # 過去 10 天
  future = 3  # 未來 4 天
  learning_rate = 0.001
  batch_size = 32   # 批次訓練
  epochs = 15       # 循環次數
  LSTM(32, L2(0.001))
  Dropout(0.25)
  patience=3  
  keras.losses.LogCosh()
"""

# 使用範例
#uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
#zip_fname = "jena_climate_2009_2016.csv.zip"
csv_fname = "stock_day_2330_2010-2024.csv"

def load_data(csv_fname):
    # 讀取 CSV 文件
    df = pd.read_csv(csv_fname)
    # Convert numeric columns by removing commas and converting to integers
    #df.drop(["成交股數", "成交金額"], axis=1, inplace=True) # inplace=True 表示不用 asign
    df.columns = ["Date","Deal", "Amount", "Open", "High", "Low", "Close", "RD", "Volumn"]
    #日期,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數

    df.set_index(["Date"], inplace=True)
    # 将指定列转换为数字类型（例如，将 'column_name' 列转换为数字）
    #df["Open"] = pd.to_numeric(df["Open"], errors='coerce')
    # 如果要将多个列转换为数字类型，可以使用以下方法：
    df[["Deal", "Amount", "Open", "High", "Low", "Close", "RD", "Volumn" ]] = df[["Deal", "Amount", "Open", "High", "Low", "Close", "RD", "Volumn" ]].apply(pd.to_numeric, errors='coerce')
    return df

# 讀取 CSV 文件
#df = pd.download_and_extract_data(uri, zip_fname, csv_fname)
df = load_data(csv_fname)
df.columns # Index(['日期', '成交股數', '成交金額', '開盤價', '最高價', '最低價', '收盤價', '漲跌價差', '成交筆數'], dtype='object')
#df = df.dropna(axis=0)
df.describe()
print(df.shape) # (120, 9)
print(df.isna().sum())
print(df.head())

# 外部聯接所有 DataFrame
#combined_df = pd.concat([df1, df2["收盤價"]], axis=1, join='outer')
# 使用最近的有效值填充缺失值
#combined_df.ffill(inplace=True)  # 向前填充
#combined_df.bfill(inplace=True)  # 向後填充

titles = [
  "Deal", "Amount", "Open", "High", "Low", "Close", "RD", "Volumn"
]

feature_keys = [
   "Deal", "Amount", "Open", "High", "Low", "Close", "RD", "Volumn"
]

colors = [
    "blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan",
]

date_time_key = "Date"

def show_raw_visualization(data):
    #time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        #t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()

# Raw Data Visualization (顯示資料分布)
#show_raw_visualization(df)


"""
2024/09/04
https://www.tensorflow.org/tutorials?hl=zh-tw
https://www.tensorflow.org/guide?hl=zh-tw
https://www.tensorflow.org/guide/keras/sequential_model
https://www.tensorflow.org/tutorials/structured_data/time_series
"""

split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0])) # 300,693 訓練比數
step = 1    # 每筆 1 天
past = 20  # 過去 10 天
future = 3  # 未來 4 天
learning_rate = 0.001
batch_size = 32   # 批次訓練 32
epochs = 15       # 循環次數 15

# 選定參數 => 蒐集資料
print(
    "選定參數:",
    ", ".join([titles[i] for i in [0, 1, 2, 3, 4, 5, 6, 7]]),
)
selected_features = [feature_keys[i] for i in [0, 1, 2, 3, 4, 5, 6, 7]]
features = df[selected_features]   # 仍然是 DataFrame
#features.index = df[date_time_key] # 定義 index
print("features 大小 => ", features.shape) # (420551, 7)
print("features 型態 => ", type(features)) # DataFrame
backup = features

# 這個過程的數學意義是將不同範圍和尺度的特徵標準化，使得每個特徵的均值為0，標準差為1，從而消除特徵之間量級差異對模型訓練的影響。
close_mean = 0  # 152.36664077669903
close_std = 0   # 74.53149165505323
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0) # 一維 (7,) ndarray
    data_std = data[:train_split].std(axis=0)   # 一維 (7,) ndarray
    return (data - data_mean) / data_std, data_mean[5], data_std[5]

def denormalize(data, mean, std):
    return (data * std) + mean

features = pd.DataFrame(features) # 此時欄位已經丟失原始資訊 => RangeIndex(0,1,2..N)

# 正規化 (僅有 .715 )
#print("features-origin", features.shape, type(features))     # DataFrame(420551, 7) 原始資料
features, close_mean, close_std = normalize(features.values, train_split)
#print("features-normalize1", features.shape, type(features)) # ndarray(420551, 7)   正規化 0~1
features = pd.DataFrame(features) # 此時欄位已經丟失原始資訊 => RangeIndex(0,1,2..N)
#print("features-normalize2", features.shape, type(features)) # DataFrame(420551, 7) 正規化 0~1


# 切割前半訓練資料, 後半驗證資料
# 注意!! pandas.loc[start:end] 是 包含 end 这个索引的
train_data = features.loc[0 : train_split - 1]  # (85, 6)
val_data = features.loc[train_split:]           # (35, 6)

# Training dataset
# 720 timestamps (720/6=120 hours)
# The training dataset labels starts from the 792nd observation (720 + 72).
start = past + future     # 792     = 720 + 72
end = start + train_split # 301485

# i => [0, 1, 2, 3, 4, 5, 6]
x_train = train_data[[i for i in range(8)]].values # (300693, 7) ndarray
y_train = features.iloc[start:end][[5]]            # (300693, 1) DataFrame (錯位 offsef 驗證用) 取第二欄 i=1

sequence_length = int(past / step)                 # 120

# 函數參考: https://keras.io/api/data_loading/timeseries/
dataset_train = timeseries_dataset_from_array(
    data    = x_train,   # 這是輸入數據的數組或數據框，包含時間序列數據點（如溫度、股價等）。
    targets = y_train,   # 這是目標數據的數組或數據框，通常與 x_train 對應。這個數據集用來表示模型需要預測的值（如未來的溫度或股價）。如果要生成無監督學習的輸入數據，可以將此參數設為 None。
    sequence_length = sequence_length, # 每個輸入序列的長度，即模型每次訓練時看多少個連續的時間步。例如，如果你設置 sequence_length=10，每個輸入序列將包含 10 個連續的數據點。
    sampling_rate = step,    # 這個參數決定了序列中兩個數據點之間的間隔。默認為 1，表示使用每個數據點。如果設置為 2，則每兩個數據點取一個樣本。
    batch_size = batch_size, # 每次訓練中使用的樣本數，即一次訓練所使用的時間序列數據的批量大小。這個參數影響到訓練時的記憶體消耗和梯度更新頻率。
) # 輸出 => 型態 _BatchDataset, 長度 1172 (約 300693 / 256 亂猜的)

# 確定驗證數據的結束點
# 這行代碼計算驗證數據的結束點，從 val_data 的長度中減去 past 和 future。
# 這樣確保驗證數據不包含最後的 792 行，因為這些行沒有對應的標籤數據。
x_end = len(val_data) - past - future # 119066=119858-792

# 確定驗證標籤的起始點
# 計算驗證標籤的起始點。這是從訓練數據分割點開始，再向後推移 past + future，以確保驗證標籤數據與驗證輸入數據正確對應。
label_start = train_split + past + future

# 提取驗證數據和標籤
x_val = val_data.iloc[:x_end][[i for i in range(8)]].values  # x_val 提取驗證數據的輸入特徵。從 val_data 中提取 x_end 行的數據，並選擇前 7 個特徵。
y_val = features.iloc[label_start:][[5]]        # y_val 則提取驗證數據的目標標籤，從 features 中選擇從 label_start 開始的數據。

# 創建驗證數據集
# 使用 timeseries_dataset_from_array 函數來生成驗證數據集。這個數據集包含了輸入序列和對應的目標標籤，
# 並會根據 sequence_length 和 sampling_rate 來創建一系列時間序列樣本。
dataset_val = timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

# 打印輸入和目標形狀 (確認資料沒問題,還沒訓練 )
for batch in dataset_train.take(1):
    # 從訓練數據集中提取一個批次，並打印其輸入 (inputs) 和目標 (targets) 的形狀。這樣可以確認數據集的結構是否符合預期。
    inputs, targets = batch

#總結
#這段代碼的主要目的是構建一個驗證數據集，其中包括經過正確預處理的時間序列數據和對應的標籤。通過確保不使用缺少標籤的數據點，
# 這段代碼幫助模型更準確地評估在看不見的數據上的表現。
print("Input shape:", inputs.numpy().shape)     # (256, 120, 7)
print("Target shape:", targets.numpy().shape)   # (256, 1)

# Training
# 這段代碼使用 Keras 建立和編譯了一個簡單的 LSTM 模型，用於時間序列預測。以下是每個步驟的詳細解釋：
# 1. 輸入層 (Input Layer)
inputs = keras.layers.Input(  #  keras.layers.Input: 這是 Keras 的輸入層，用於定義模型的輸入形狀。
    shape=(inputs.shape[1],   #  shape=(inputs.shape[1], inputs.shape[2]): inputs.shape 是輸入數據的形狀，通常是 (batch_size, sequence_length, num_features)。
           inputs.shape[2]))  # inputs.shape[1] 代表序列的長度（時間步數），inputs.shape[2] 代表每個時間步的特徵數量。

# 2. LSTM 層 (LSTM Layer)
#lstm_out = keras.layers.LSTM(32)( #  keras.layers.LSTM(32): 這是一個 LSTM 層，用於處理時間序列數據。32 表示這個 LSTM 層有 32 個單元（units）。
lstm_out = keras.layers.LSTM(32, kernel_regularizer=keras.regularizers.l2(0.001))( # AI 建議
    inputs)  #   (inputs): 這意味著這個 LSTM 層接收來自輸入層的數據。(Ben: chatgpt 說只是參考結構)

# AI 建議
#lstm_out = keras.layers.Dropout(0.25)(lstm_out)  # 新增 Dropout 層來減少過擬合

# 3. 全連接層 (Dense Layer)
#   keras.layers.Dense(1): 這是一個全連接層，輸出一個標量值。這裡 1 表示該層有一個神經元。
outputs = keras.layers.Dense(1)(lstm_out) #   (lstm_out): 這意味著這個全連接層接收來自 LSTM 層的輸出。

# 4. 建立模型 (Model Creation)
model = keras.Model( # keras.Model: 這是用於定義模型的類。你需要指定模型的輸入和輸出層。
    inputs=inputs,   # inputs=inputs: 這裡的 inputs 代表模型的輸入層。
    outputs=outputs) # outputs=outputs: 這裡的 outputs 代表模型的輸出層。

# 5. 編譯模型 (Compile Model)
#model.compile(
#    optimizer=keras.optimizers.Adam( #  optimizer=keras.optimizers.Adam(learning_rate=learning_rate): 這裡使用了 Adam 優化器來更新模型的權重。
#    learning_rate=learning_rate), # learning_rate 是學習率，控制每次更新權重時步伐的大小。
#    loss="mse") #  loss="mse": 指定損失函數為均方誤差（Mean Squared Error, MSE）。這是一個常用於回歸問題的損失函數，衡量預測值與實際值之間的平方差。

# 使用 mae 損失 (4.15 => 2.91)
#model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mae")
# 使用 Huber 損失
#model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.Huber())
# 使用 LogCosh 損失
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.LogCosh())

# 6. 模型總結 (Model Summary)
#  model.summary(): 打印模型的結構，包括每層的名稱、輸出形狀以及參數數量。這對於檢查模型的結構是否符合預期非常有用。
model.summary()

# 這段代碼使用了 Keras 的回調函數來管理模型的訓練過程，具體包括保存檢查點（Checkpoints）和提前停止（Early Stopping）機制。以下是每個部分的詳細解釋：
# 1. 檢查點文件路徑 (Checkpoint Path)
path_checkpoint = "model_checkpoint.weights.h5" #   path_checkpoint: 這是一個字串，表示保存模型權重的文件路徑。在訓練過程中，模型的最佳權重會被保存到這個路徑。

# 2. 提前停止回調 (EarlyStopping Callback)
es_callback = keras.callbacks.EarlyStopping( # keras.callbacks.EarlyStopping: 這個回調函數用於在驗證損失（val_loss）不再改進時提前終止訓練。
    monitor="val_loss",  #  monitor="val_loss": 這表示我們監控的是驗證損失（val_loss）。
    min_delta=0,  # min_delta=0: 訓練會在驗證損失不再改進超過這個最小變化量時停止。
    #restore_best_weights=True,
    patience=3)   # patience=5: 如果驗證損失在 5 個訓練週期（epochs）內沒有改進，訓練就會被停止。

# 3. 模型檢查點回調 (ModelCheckpoint Callback)
# keras.callbacks.ModelCheckpoint: 這個回調函數用於在訓練過程中定期保存模型的權重。
modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",       # monitor="val_loss": 同樣監控驗證損失（val_loss）。
    filepath=path_checkpoint, # filepath=path_checkpoint: 保存模型權重的文件路徑。
    verbose=1,  #  verbose=1: 設定為 1 會在每次保存權重時打印出一條信息，說明已經保存了檢查點。
    save_weights_only=True,   # save_weights_only=True: 只保存模型的權重，而不保存整個模型結構。
    save_best_only=True,      # save_best_only=True: 這意味著只有當模型的驗證損失優於之前的最佳結果時，才會保存模型權重。
)

# 4. 模型訓練 (Model Training)
history = model.fit( # model.fit(): 這是 Keras 中用於訓練模型的主要方法。
    dataset_train,  # dataset_train: 訓練數據集。
    epochs=epochs,  # epochs=epochs: 訓練的總週期數，這裡的 epochs 是一個變數，代表訓練的迭代次數。
    validation_data=dataset_val, # validation_data=dataset_val: 驗證數據集，用於在訓練過程中評估模型的表現。
    callbacks=[es_callback, modelckpt_callback], # callbacks=[es_callback, modelckpt_callback]: 訓練過程中使用的回調函數列表，包括提前停止和模型檢查點回調。
)


def visualize_loss(history, title):
    # 2. 提取損失數據 (Extracting Loss Data)
    loss = history.history["loss"]         # - loss: 這是訓練損失的列表，從 history.history["loss"] 中提取出來。這個列表中記錄了每一個訓練週期（epoch）的損失值。
    val_loss = history.history["val_loss"] # - val_loss: 這是驗證損失的列表，從 history.history["val_loss"] 中提取出來。它記錄了每個訓練週期後模型在驗證數據上的損失。
    # 3. 定義訓練週期 (Define Epochs)
    epochs = range(len(loss)) # epochs: 這定義了一個範圍對象，表示每個訓練週期的序列。這裡的範圍從 0 開始，到 len(loss) 結束，對應於每個訓練週期的損失值。
    # 4. 創建圖表 (Create Plot)
    plt.figure()
    # 5. 繪製損失曲線 (Plot Loss Curves)
    plt.plot(epochs, loss, "b", label="Training loss")       # 繪製訓練損失曲線，"b" 表示曲線的顏色為藍色，label="Training loss" 設置圖例中的標籤。
    plt.plot(epochs, val_loss, "r", label="Validation loss") # 繪製驗證損失曲線，"r" 表示曲線的顏色為紅色，label="Validation loss" 設置圖例中的標籤。
    # 6. 添加標題和標籤 (Add Title and Labels)
    plt.title(title)      # title: 圖表的標題，用於說明可視化的內容。
    plt.xlabel("Epochs")  # 設置 X 軸的標籤為“Epochs”，表示訓練週期。
    plt.ylabel("Loss")    # 設置 Y 軸的標籤為“Loss”，表示損失值。
    plt.legend()          #  顯示圖例，區分訓練損失和驗證損失。
    plt.show()

visualize_loss(history, "Training and Validation Loss")



def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]  # labels: 設定三個標籤，分別是“歷史數據”、“真實未來值”和“模型預測”。
    marker = [".-", "rx", "go"]  # marker: 設定三種標記樣式，分別對應上面三個標籤。
    time_steps = list(range(-(plot_data[0].shape[0]), 0))  # time_steps: 創建一個時間步的列表，範圍是從 -N 到 0，這裡的 N 是歷史數據的長度。
    if delta:
        future = delta       # future: 如果 delta 被指定，則 future 設置為 delta；否則，設置為 0。
    else:
        future = 0
    # 2. 繪製圖表
    plt.title(title)  # 設定圖表的標題。
    for i, val in enumerate(plot_data): # 迭代 plot_data 列表，i 是索引，val 是對應的數據值。
        if i: # 如果 i 不等於 0，則繪製 future（預測的時間步）的數據，並使用不同的標記樣式。
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
            predicted_values = denormalize(plot_data[i], close_mean, close_std)
            print(i, predicted_values)
        else: # 如果 i 等於 0，則繪製歷史數據。
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend() # 顯示圖例，對應每個標籤。
    plt.xlim([time_steps[0], (future + 5) * 2])  # 設置 X 軸的顯示範圍，從 time_steps[0] 到 (future + 5)
    plt.xlabel("Time-Step")
    plt.show()
    return

# 3. 預測和可視化
for x, y in dataset_val.take(5): # 這段代碼從驗證數據集中取得 5 個批次的數據，每個批次包含特徵 x 和標籤 y。
    show_plot( # 調用 show_plot 函數來顯示每個批次的預測結果：
        # 取得批次中第一個樣本的第二個特徵值（這裡假設是與溫度相關的特徵），並轉換為 NumPy 陣列。
        # y[0].numpy() 取得批次中第一個樣本的真實標籤值，並轉換為 NumPy 陣列。
        [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],   # model.predict(x)[0]: 使用模型對 x 進行預測，並取得預測值的第一個結果。
        3, # 設置為 12，表示未來 12 個時間步的預測。
        "Single Step Prediction",
    )

import numpy as np

val_mean = backup[train_split:].mean(axis=0) # 一維 (7,) ndarray
val_std = backup[train_split:].std(axis=0)   # 一維 (7,) ndarray
print(val_mean, val_std)

# 把 dataset_val 的數據提取出來
x_val_np = np.concatenate([x for x, _ in dataset_val], axis=0)

# 對 x_val_np 進行預測
predictions = model.predict(x_val_np)

#k = denormalize(predictions[0:1][0], val_mean[5], val_std[5])
print(predictions[0:1] * val_std[5] + val_mean[5])
