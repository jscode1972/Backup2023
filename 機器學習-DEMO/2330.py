from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import timeseries_dataset_from_array

# 使用範例
#uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
#zip_fname = "jena_climate_2009_2016.csv.zip"
csv_fname = "stock_day_2330_2010-2024.csv"

#def download_and_extract_data(uri, zip_fname, csv_fname):
#    # 下載 zip 文件
#    zip_path = keras.utils.get_file(origin=uri, fname=zip_fname)
#    # 解壓縮 zip 文件
#    with ZipFile(zip_path, 'r') as zip_file:
#        zip_file.extractall()
#    # 讀取 CSV 文件
#    df = pd.read_csv(csv_fname)
#    return df

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
