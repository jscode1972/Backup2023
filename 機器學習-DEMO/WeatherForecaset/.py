# 官網範例 https://keras.io/examples/timeseries/timeseries_weather_forecasting/
# colab  https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/timeseries/ipynb/timeseries_weather_forecasting.ipynb
# 資料載點 "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

from zipfile import ZipFile
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.utils import timeseries_dataset_from_array, timeseries_dataset_from_array

# 使用範例
uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_fname = "jena_climate_2009_2016.csv.zip"
csv_fname = "jena_climate_2009_2016.csv"

def download_and_extract_data(uri, zip_fname, csv_fname):
    # 下載 zip 文件
    zip_path = keras.utils.get_file(origin=uri, fname=zip_fname)
    
    # 解壓縮 zip 文件
    with ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall()
    
    # 讀取 CSV 文件
    df = pd.read_csv(csv_fname)
    return df

def load_data(csv_fname):
    # 讀取 CSV 文件
    df = pd.read_csv(csv_fname)
    return df   

# 讀取 CSV 文件
#df = pd.download_and_extract_data(uri, zip_fname, csv_fname)
df = load_data(csv_fname)
print(df.shape) # (420551, 15)


titles = [
    "Pressure",
    "Temperature",
    "Temperature in Kelvin",
    "Temperature (dew point)",
    "Relative Humidity",
    "Saturation vapor pressure",
    "Vapor pressure",
    "Vapor pressure deficit",
    "Specific humidity",
    "Water vapor concentration",
    "Airtight",
    "Wind speed",
    "Maximum wind speed",
    "Wind direction in degrees",
]

feature_keys = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

date_time_key = "Date Time"

def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
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
# show_raw_visualization(df)


# Data Preprocessing
# 資料選取：從資料集中選取約 300,000 個數據點用於訓練。因為每 10 分鐘記錄一次觀測值（每小時 6 次），認為在 60 分鐘內變化不大，因此每小時取一個樣本點。
# 時間序列追蹤：使用過去 720 個時間點的數據（相當於 120 小時或 5 天）來預測 72 個時間點（12 小時）後的溫度。
# 數據正規化：由於每個特徵的值範圍不同，在訓練神經網路前，通過減去平均值並除以標準差來將特徵值正規化到 [0, 1] 範圍內。
# 資料分割：71.5% 的數據（約 300,693 行）用於訓練模型，可以通過調整 split_fraction 參數來改變此比例。
# 訓練樣本和標籤：模型會看到前 5 天的資料（720 個每小時取樣的觀測值），並使用接下來 12 小時（72 個觀測值）後的溫度作為預測標籤。
split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0])) # 300,693 筆 = 0.715 * 420,551
step = 6     # 每十分量一次, 一小時 6 次
past = 720   # 過去 720次量測點 => 720/6=120小時 => 5天
future = 72
learning_rate = 0.001
batch_size = 256
epochs = 10

# 蒐集資料
print(
    "選定參數:",
    ", ".join([titles[i] for i in [0, 1, 5, 7, 8, 10, 11]]),
)
selected_features = [feature_keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]
features = df[selected_features]   # 仍然是 DataFrame
features.index = df[date_time_key]
#print("features", type(features)) # 仍然是 DataFrame


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

# 正規化
features = normalize(features.values, train_split)
features = pd.DataFrame(features)
#print(features.head())

# 切割前半訓練資料, 後半驗證資料
train_data = features.loc[0 : train_split - 1]  # (300693, 7)
val_data = features.loc[train_split:]           # (119858, 7)

# Training dataset 
# 720 timestamps (720/6=120 hours)
# The training dataset labels starts from the 792nd observation (720 + 72).
start = past + future     # 792     = 720 + 72 
end = start + train_split # 301485

# i => [0, 1, 2, 3, 4, 5, 6] 
x_train = train_data[[i for i in range(7)]].values # (300693, 7) ndarray
y_train = features.iloc[start:end][[1]]            # (300693, 1) DataFrame

sequence_length = int(past / step)                 # 120

"""
這段代碼使用了 keras.preprocessing.timeseries_dataset_from_array 函數來從時序數據中生成用於訓練的數據集。以下是這段代碼的詳細解釋：
timeseries_dataset_from_array 函數適用於時間序列數據的預處理。時間序列數據是按時間順序排列的數據點（如每小時溫度、每日股價等），這些數據通常按固定的時間間隔收集。
工作原理
  這段代碼的目的在於將原始的時間序列數據 x_train 和對應的目標值 y_train 分割成多個序列，每個序列包含 sequence_length 個數據點。每個序列之間的間隔由 sampling_rate 決定。這些序列將被分批 (batch_size) 輸入到模型中進行訓練。
實際效果
  生成的 dataset_train 將是一個 TensorFlow 的 Dataset 對象，其中包含從原始數據中提取的許多批次（batch），每個批次包含多個長度為 sequence_length 的子序列。這些子序列將被用於訓練模型，使模型能夠學習時間序列的模式，進而對未來的數據點進行預測。
"""
dataset_train = timeseries_dataset_from_array(
    x_train,   # 這是輸入數據的數組或數據框，包含時間序列數據點（如溫度、股價等）。
    y_train,   # 這是目標數據的數組或數據框，通常與 x_train 對應。這個數據集用來表示模型需要預測的值（如未來的溫度或股價）。如果要生成無監督學習的輸入數據，可以將此參數設為 None。
    sequence_length=sequence_length, # 每個輸入序列的長度，即模型每次訓練時看多少個連續的時間步。例如，如果你設置 sequence_length=10，每個輸入序列將包含 10 個連續的數據點。
    sampling_rate=step,    # 這個參數決定了序列中兩個數據點之間的間隔。默認為 1，表示使用每個數據點。如果設置為 2，則每兩個數據點取一個樣本。
    batch_size=batch_size, # 每次訓練中使用的樣本數，即一次訓練所使用的時間序列數據的批量大小。這個參數影響到訓練時的記憶體消耗和梯度更新頻率。
) # _BatchDataset


"""
# Validation dataset
這段代碼用於創建驗證數據集，它的目的是確保驗證數據集不包含時間序列中最後的 792 行，因為這些行缺少對應的標籤數據。以下是這段代碼的詳細解說：
背景
  在時間序列預測中，我們通常會將數據分為訓練集和驗證集。驗證集用來評估模型的表現。然而，由於某些時間序列數據缺乏標籤（例如未來的數據無法提前獲得），我們需要小心處理這些數據。
關鍵參數解釋
  past: 這個變數表示模型使用過去多少個時間點的數據來進行預測。
  future: 這個變數表示模型要預測的未來時間點的數量。
  train_split: 這是訓練數據和驗證數據之間的分割點。
"""

# 確定驗證數據的結束點
# 這行代碼計算驗證數據的結束點，從 val_data 的長度中減去 past 和 future。這樣確保驗證數據不包含最後的 792 行，因為這些行沒有對應的標籤數據。
x_end = len(val_data) - past - future 

# 確定驗證標籤的起始點
# 計算驗證標籤的起始點。這是從訓練數據分割點開始，再向後推移 past + future，以確保驗證標籤數據與驗證輸入數據正確對應。
label_start = train_split + past + future

# 提取驗證數據和標籤
x_val = val_data.iloc[:x_end][[i for i in range(7)]].values  # x_val 提取驗證數據的輸入特徵。從 val_data 中提取 x_end 行的數據，並選擇前 7 個特徵。
y_val = features.iloc[label_start:][[1]]        # y_val 則提取驗證數據的目標標籤，從 features 中選擇從 label_start 開始的數據。

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

# 打印輸入和目標形狀
for batch in dataset_train.take(1):
    # 從訓練數據集中提取一個批次，並打印其輸入 (inputs) 和目標 (targets) 的形狀。這樣可以確認數據集的結構是否符合預期。
    inputs, targets = batch 

#總結
#這段代碼的主要目的是構建一個驗證數據集，其中包括經過正確預處理的時間序列數據和對應的標籤。通過確保不使用缺少標籤的數據點，
# 這段代碼幫助模型更準確地評估在看不見的數據上的表現。
print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)


# Training
# 這段代碼使用 Keras 建立和編譯了一個簡單的 LSTM 模型，用於時間序列預測。以下是每個步驟的詳細解釋：
# 1. 輸入層 (Input Layer)
inputs = keras.layers.Input(  #  keras.layers.Input: 這是 Keras 的輸入層，用於定義模型的輸入形狀。
    shape=(inputs.shape[1],   #  shape=(inputs.shape[1], inputs.shape[2]): inputs.shape 是輸入數據的形狀，通常是 (batch_size, sequence_length, num_features)。
           inputs.shape[2]))  # inputs.shape[1] 代表序列的長度（時間步數），inputs.shape[2] 代表每個時間步的特徵數量。

# 2. LSTM 層 (LSTM Layer)
lstm_out = keras.layers.LSTM(32)( #  keras.layers.LSTM(32): 這是一個 LSTM 層，用於處理時間序列數據。32 表示這個 LSTM 層有 32 個單元（units）。
    inputs)  #   (inputs): 這意味著這個 LSTM 層接收來自輸入層的數據。

# 3. 全連接層 (Dense Layer)
#   keras.layers.Dense(1): 這是一個全連接層，輸出一個標量值。這裡 1 表示該層有一個神經元。
outputs = keras.layers.Dense(1)(lstm_out) #   (lstm_out): 這意味著這個全連接層接收來自 LSTM 層的輸出。

# 4. 建立模型 (Model Creation)
model = keras.Model( # keras.Model: 這是用於定義模型的類。你需要指定模型的輸入和輸出層。
    inputs=inputs,   # inputs=inputs: 這裡的 inputs 代表模型的輸入層。
    outputs=outputs) # outputs=outputs: 這裡的 outputs 代表模型的輸出層。

# 5. 編譯模型 (Compile Model)
model.compile(
    optimizer=keras.optimizers.Adam( #  optimizer=keras.optimizers.Adam(learning_rate=learning_rate): 這裡使用了 Adam 優化器來更新模型的權重。
    learning_rate=learning_rate), # learning_rate 是學習率，控制每次更新權重時步伐的大小。
    loss="mse") #  loss="mse": 指定損失函數為均方誤差（Mean Squared Error, MSE）。這是一個常用於回歸問題的損失函數，衡量預測值與實際值之間的平方差。

# 6. 模型總結 (Model Summary)
#  model.summary(): 打印模型的結構，包括每層的名稱、輸出形狀以及參數數量。這對於檢查模型的結構是否符合預期非常有用。
model.summary()

""" 總結
  這段代碼建立了一個簡單的 LSTM 模型，該模型接收時間序列數據並預測一個標量值。模型由三層組成：一個輸入層，一個具有 32 個單元的 LSTM 層，和一個輸出標量的全連接層。模型編譯時使用了 Adam 優化器和均方誤差損失函數。
"""


# 這段代碼使用了 Keras 的回調函數來管理模型的訓練過程，具體包括保存檢查點（Checkpoints）和提前停止（Early Stopping）機制。以下是每個部分的詳細解釋：
# 1. 檢查點文件路徑 (Checkpoint Path)
path_checkpoint = "model_checkpoint.weights.h5" #   path_checkpoint: 這是一個字串，表示保存模型權重的文件路徑。在訓練過程中，模型的最佳權重會被保存到這個路徑。

# 2. 提前停止回調 (EarlyStopping Callback)
es_callback = keras.callbacks.EarlyStopping( # keras.callbacks.EarlyStopping: 這個回調函數用於在驗證損失（val_loss）不再改進時提前終止訓練。
    monitor="val_loss",  #  monitor="val_loss": 這表示我們監控的是驗證損失（val_loss）。
    min_delta=0,  # min_delta=0: 訓練會在驗證損失不再改進超過這個最小變化量時停止。
    patience=5)   # patience=5: 如果驗證損失在 5 個訓練週期（epochs）內沒有改進，訓練就會被停止。

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

""" 總結  
  這段代碼使用 EarlyStopping 回調來避免過度擬合，當驗證損失在指定的 patience（耐心）期內不再改進時，訓練將會停止。同時，ModelCheckpoint 回調會在訓練過程中持續保存最佳模型的權重，確保最優模型權重在訓練結束後可用。這些回調函數可以大幅提高模型訓練的效率和效果。
"""


# 這段代碼定義了一個函數來可視化訓練和驗證過程中的損失變化。以下是每個部分的詳細解釋：
# 1. 函數定義 (visualize_loss)
#   visualize_loss: 這是一個函數，用於繪製模型訓練過程中的損失曲線。
#   history: 這是 model.fit() 返回的訓練歷史對象，包含訓練和驗證過程中的各種指標，例如損失和準確度。
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

# 這段程式碼展示了如何使用已訓練好的模型來進行預測，並將模型的預測結果與真實值進行可視化比較。以下是每一部分的詳細說明：
# 1. 定義 show_plot 函數
# - plot_data: 這是一個包含歷史數據、真實未來值和模型預測值的列表。
# - delta: 用來指定預測的時間步數偏移（即未來多少個時間步的預測）。
# - title: 圖表的標題，用於說明可視化的內容。
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
        12, # 設置為 12，表示未來 12 個時間步的預測。
        "Single Step Prediction",
    )
""" 總結
這段代碼展示了如何使用訓練好的模型對驗證數據集進行預測，並將預測結果與真實值進行可視化比較。
這種可視化可以幫助你直觀地檢查模型的預測性能，特別是在時序數據的情境下。你可以看到模型在歷史數據基礎上做出的預測與實際未來數據的吻合度。
"""


