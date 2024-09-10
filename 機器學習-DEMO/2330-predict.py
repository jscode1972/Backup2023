import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from zipfile import ZipFile
from sklearn.preprocessing import MinMaxScaler
from keras.utils import timeseries_dataset_from_array

# 參考範例  "jena_climate_2009_2016.csv.zip"
#uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
csv_fname = "stock_day_2330_2010-2024.csv"
split_fraction = 0.8 # 提高訓練比有效
#train_split = int(split_fraction * int(df.shape[0])) # 300,693 訓練比數
step = 1    # 每筆 1 天
past = 20   # 過去 10 天
future = 3  # 未來 4 天
learning_rate = 0.001
batch_size = 32   # 批次訓練 32
epochs = 5       # 循環次數 15
y_index = 5       # close
scaler_X = MinMaxScaler()  # 用于特征数据 (X)
scaler_y = MinMaxScaler()  # 用于目标数据 (y)

def load_data(csv_fname):
    # 讀取 CSV 文件
    df = pd.read_csv(csv_fname)
    # Convert numeric columns by removing commas and converting to integers
    #df.drop(["成交股數", "成交金額"], axis=1, inplace=True) # inplace=True 表示不用 asign
    #日期,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數
    feature_columns = ["Date", "Deal", "Amount", "Open", "High", "Low", "Close", "RD", "Volumn"]
    df.columns = feature_columns
    #features.index = df[date_time_key] # 定義 index
    df.set_index(["Date"], inplace=True)
    # 将指定列转换为数字类型（例如，将 'column_name' 列转换为数字）
    #df["Open"] = pd.to_numeric(df["Open"], errors='coerce')
    # 如果要将多个列转换为数字类型，可以使用以下方法：
    feature_columns.remove("Date")
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
    return df

def split_data(df, split_fraction):
    # 切割前半訓練資料, 後半驗證資料
    train_split = int(split_fraction * int(df.shape[0])) # 300,693 訓練比數
    # 注意!! pandas.loc[start:end] 是 包含 end 这个索引的
    train_data = df.iloc[0:train_split]    # (2881, 8)
    val_data   = df.iloc[train_split:]     # (721, 8)
    return train_data, val_data

def prepare_data(train_data, val_data, past, future, indice):
    # result = pd.concat([train_data.iloc[3+2-1:, [5,6]], val_data.iloc[:3+2-1, [5,6]]], axis=0, ignore_index=True)
    X_train = train_data
    y_train1 = train_data.iloc[past+future-1:,[5]]
    y_train2 = val_data.iloc[:past+future-1,[5]]
    y_train = pd.concat([y_train1, y_train2], axis=0, ignore_index=True)
    #y_train = y_train.values.reshape(-1, 1)
    # 計算驗證標籤的起始點。這是從訓練數據分割點開始，再向後推移 past + future，以確保驗證標籤數據與驗證輸入數據正確對應。
    # 提取驗證數據和標籤
    X_val = val_data.iloc[:len(val_data)-past-future+1]
    y_val = val_data.iloc[(past+future-1):,[5]]
    #y_val = y_val.values.reshape(-1, 1)
    return  X_train, y_train, X_val, y_val

def normalize_data(X_train, y_train, X_val, y_val):
    # 创建 MinMaxScaler 实例
    scaler_X.fit(X_train)
    scaler_y.fit(y_train) # values.reshape(-1, 1)
    # 正規化 .values.reshape(-1, 1)
    X_train = scaler_X.transform(X_train)
    y_train = scaler_y.transform(y_train)
    X_val   = scaler_X.transform(X_val)
    y_val   = scaler_y.transform(y_val)
    return X_train, y_train, X_val, y_val

def create_dataset(X_train, y_train, X_val, y_val, past, step, batch_size):
    sequence_length = int(past / step) # 120
    # 函數參考: https://keras.io/api/data_loading/timeseries/
    dataset_train = timeseries_dataset_from_array(
        data    = X_train,   # 這是輸入數據的數組或數據框，包含時間序列數據點（如溫度、股價等）。
        targets = y_train,   # 這是目標數據的數組或數據框，通常與 x_train 對應。這個數據集用來表示模型需要預測的值（如未來的溫度或股價）。如果要生成無監督學習的輸入數據，可以將此參數設為 None。
        sequence_length = sequence_length, # 每個輸入序列的長度，即模型每次訓練時看多少個連續的時間步。例如，如果你設置 sequence_length=10，每個輸入序列將包含 10 個連續的數據點。
        sampling_rate = step,    # 這個參數決定了序列中兩個數據點之間的間隔。默認為 1，表示使用每個數據點。如果設置為 2，則每兩個數據點取一個樣本。
        batch_size = batch_size, # 每次訓練中使用的樣本數，即一次訓練所使用的時間序列數據的批量大小。這個參數影響到訓練時的記憶體消耗和梯度更新頻率。
    ) # 輸出 => 型態 _BatchDataset, 長度 1172 (約 300693 / 256 亂猜的)
    # 創建驗證數據集
    # 使用 timeseries_dataset_from_array 函數來生成驗證數據集。這個數據集包含了輸入序列和對應的目標標籤，
    # 並會根據 sequence_length 和 sampling_rate 來創建一系列時間序列樣本。
    dataset_val = timeseries_dataset_from_array(
        X_val,
        y_val,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
    # 打印輸入和目標形狀 (確認資料沒問題,還沒訓練 )
    for batch in dataset_train.take(1):
        # 從訓練數據集中提取一個批次，並打印其輸入 (inputs) 和目標 (targets) 的形狀。這樣可以確認數據集的結構是否符合預期。
        inputs, targets = batch
    return dataset_train, dataset_val, inputs, targets

def build_model(inputs):
    # Training
    # 這段代碼使用 Keras 建立和編譯了一個簡單的 LSTM 模型，用於時間序列預測。以下是每個步驟的詳細解釋：
    # 1. 輸入層 (Input Layer)
    inputs = keras.layers.Input(  #  keras.layers.Input: 這是 Keras 的輸入層，用於定義模型的輸入形狀。
        shape=(inputs.shape[1],   #  shape=(inputs.shape[1], inputs.shape[2]): inputs.shape 是輸入數據的形狀，通常是 (batch_size, sequence_length, num_features)。
               inputs.shape[2]))  # inputs.shape[1] 代表序列的長度（時間步數），inputs.shape[2] 代表每個時間步的特徵數量。

    # 2. LSTM 層 (LSTM Layer)
    #lstm_out = keras.layers.LSTM(32)( #  keras.layers.LSTM(32): 這是一個 LSTM 層，用於處理時間序列數據。32 表示這個 LSTM 層有 32 個單元（units）。
    lstm_out = keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001))( # AI 建議
        inputs)  #   (inputs): 這意味著這個 LSTM 層接收來自輸入層的數據。(Ben: chatgpt 說只是參考結構)

    # 第二层 LSTM，返回最后一个时间步的输出
    # 若加此行 前面 LSTM 要補上 return_sequences=True
    #lstm_out = keras.layers.LSTM(32)(lstm_out) # 變慢而且沒幫助

    # AI 建議
    lstm_out = keras.layers.Dropout(0.3)(lstm_out)  # 新增 Dropout 層來減少過擬合
    # 亂加一層
    #lstm_out = keras.layers.Dense(10)(lstm_out)

    # 3. 全連接層 (Dense Layer)
    #   keras.layers.Dense(1): 這是一個全連接層，輸出一個標量值。這裡 1 表示該層有一個神經元。
    outputs = keras.layers.Dense(1)(lstm_out) #   (lstm_out): 這意味著這個全連接層接收來自 LSTM 層的輸出。

    # 4. 建立模型 (Model Creation)
    model = keras.Model( # keras.Model: 這是用於定義模型的類。你需要指定模型的輸入和輸出層。
        inputs=inputs,   # inputs=inputs: 這裡的 inputs 代表模型的輸入層。
        outputs=outputs) # outputs=outputs: 這裡的 outputs 代表模型的輸出層。
    return model

def training_model(model, learning_rate, dataset_train, dataset_val, epochs):
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
        patience=4)   # patience=5: 如果驗證損失在 5 個訓練週期（epochs）內沒有改進，訓練就會被停止。

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
    return history

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

"""def show_plot(plot_data, delta, title):
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
            #predicted_values = denormalize(plot_data[i], close_mean, close_std)
            #print(i, predicted_values)
        else: # 如果 i 等於 0，則繪製歷史數據。
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend() # 顯示圖例，對應每個標籤。
    plt.xlim([time_steps[0], (future + 5) * 2])  # 設置 X 軸的顯示範圍，從 time_steps[0] 到 (future + 5)
    plt.xlabel("Time-Step")
    plt.show()
    return"""

def predict_last(model, input):
    last_tran = scaler_X.transform(input)
    last_pred = model.predict(last_tran.reshape(1, 20, 8))
    print(last_pred) 
    output = [] #scaler_y.inverse_transform(last_pred)
    return output

#df = pd.download_and_extract_data(uri, zip_fname, csv_fname)
df = load_data(csv_fname)
print(df.head(1))
print(df.shape) # (120, 9)
print(df.isna().sum())
df.describe()

# 準備資料 (尚未正規化)
train_data, val_data = split_data(df, split_fraction)
print("data-shape", train_data.shape, val_data.shape) # (2881, 8) (721, 8)

# 準備訓練集/驗證集
X_train, y_train, X_val, y_val = prepare_data(train_data, val_data, past, future, 5)
print("train-shape", X_train.shape, y_train.shape) # (2881, 8) (2881, 1)
print("val-shape",   X_val.shape,   y_val.shape)   # (699, 8)  (699, 1)

# 正規化
X_train, y_train, X_val, y_val = normalize_data(X_train, y_train, X_val, y_val)
print("train-正規化", X_train.shape, y_train.shape) # (2881, 8) (721, 8)
print("val-正規化",   X_val.shape,   y_val.shape)   # (699, 8)  (699, 1)

#這段代碼的主要目的是構建一個驗證數據集，其中包括經過正確預處理的時間序列數據和對應的標籤。通過確保不使用缺少標籤的數據點，
# 這段代碼幫助模型更準確地評估在看不見的數據上的表現。
dataset_train, dataset_val, inputs, targets = create_dataset(X_train, y_train, X_val, y_val, past, step, batch_size)
print("Input shape:",  inputs.numpy().shape)         # (32, 20, 8)
print("Target shape:", targets.numpy().shape)        # (32, 1)

# 編譯模型
model = build_model(inputs)

# 開始訓練
history = training_model(model, learning_rate, dataset_train, dataset_val, epochs)

# 視覺化
visualize_loss(history, "Training and Validation Loss")

# 不是 X_val (閹割版)
print(val_data[-20:])
#x = predict_last(model, val_data[-20:])
#print("預測", x)

# 3. 預測和可視化
"""for x, y in dataset_val.take(10): # 這段代碼從驗證數據集中取得 5 個批次的數據，每個批次包含特徵 x 和標籤 y。
    show_plot( # 調用 show_plot 函數來顯示每個批次的預測結果：
        # 取得批次中第一個樣本的第二個特徵值（這裡假設是與溫度相關的特徵），並轉換為 NumPy 陣列。
        # y[0].numpy() 取得批次中第一個樣本的真實標籤值，並轉換為 NumPy 陣列。
        [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],   # model.predict(x)[0]: 使用模型對 x 進行預測，並取得預測值的第一個結果。
        3, # 設置為 12，表示未來 12 個時間步的預測。
        "Single Step Prediction",
    )"""
