import pandas as pd

# 載入Google股價的訓練資料集
df_train = pd.read_csv("GOOG_Stock_Price_Train.csv",
                       index_col="Date",parse_dates=True)
print(df_train.head())
df_train.head().to_html("Ch12_2_1.html")
