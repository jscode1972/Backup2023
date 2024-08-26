"""
Created on Tue Aug 27 00:43:16 2024
@author: wphuang
---- 優點 --------------------------------------------------------------------------------------------------------
模組化：     每個功能都被清晰地分割成獨立的方法，如 drop_columns、fill_missing_values、encode_categorical_data 等，使代碼更易於理解和維護。
數據檢查：   在 describe 方法中，提供了資料集形狀、統計摘要、數據類型和缺失值的檢查，這有助於在數據處理前對數據有全面的了解。
分類資料編碼：你對 sex 和 embarked 欄位進行了有效的數據編碼處理，將分類資料轉換為數值形式，這是機器學習模型所需要的。
目標欄位移動：move_target_to_last 方法確保了目標欄位（survived）位於 DataFrame 的最後，這對於後續的數據分割和模型訓練非常有幫助。
分割資料：   將資料集分割為訓練集和測試集（80/20），並保存為 CSV 文件，使得處理過的資料可以方便地輸入到機器學習模型中。
----- 建議 --------------------------------------------------------------------------------------------------------
異常處理：     考慮在方法中添加異常處理機制。例如，檢查讀取的 CSV 文件是否存在，或者在填補缺失值時檢查目標欄位是否存在缺失值。
文檔字符串：   為每個方法提供清晰的文檔字符串，解釋它們的功能和參數。這不僅有助於你自己未來的維護，也對於其他可能使用這段代碼的開發者非常有幫助。
欄位名稱硬編碼：目前代碼中有一些硬編碼的欄位名稱（如 "name", "ticket", "cabin", "sex", "embarked"）。考慮將這些欄位名稱定義為常量，或接受動態輸入，以提高代碼的靈活性。
更多的數據處理：如果數據集有其他需要處理的欄位或異常值，你可以考慮擴展該類別，以涵蓋更多的數據清理步驟，例如處理異常值、標準化數值欄位、生成新的特徵等。
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

class TitanicDataPreparer:
    def __init__(self, seed, data_csv):
        """初始化，載入資料集"""
        np.random.seed(seed)
        self.df = pd.read_csv(data_csv)
        print("Data loaded successfully.")
    def describe(self):
        print('--------------------- shape ------------------------')
        print(self.df.shape)      # 資料集形狀
        print('--------------------- head ------------------------')
        print(self.df.head(2))    # 看看資料
        print('--------------------- describe ------------------------')
        print(self.df.describe()) # 統計摘要
        print('--------------------- info ------------------------')
        print(self.df.info())     # 資料類型和缺失值
        print('--------------------- isnull ------------------------')
        print(self.df.isnull().sum()) # 缺失值統計
    def drop_columns(self):
        """刪除不需要的欄位"""
        self.df.drop(["name", "ticket", "cabin"], axis=1, inplace=True)
        print("Unnecessary columns dropped.")
    def fill_missing_values(self):
        print('--------------------- fill_missing_values ------------------------')
        """填補缺失值 (將遺漏欄位填入平均值)"""
        self.df[["age"]] = self.df[["age"]].fillna(value=self.df[["age"]].mean())
        self.df[["fare"]] = self.df[["fare"]].fillna(value=self.df[["fare"]].mean())
        self.df[["embarked"]] = self.df[["embarked"]].fillna(value=self.df["embarked"].value_counts().idxmax())
        print(self.df["embarked"].value_counts())
        print(self.df["embarked"].value_counts().idxmax())
    def encode_categorical_data(self):
        """將分類資料轉換為數值"""
        self.df["sex"] = self.df["sex"].map( {"female": 1, "male": 0} ).astype(int)
        # Embarked欄位的One-hot編碼
        enbarked_one_hot = pd.get_dummies(self.df["embarked"], prefix="embarked") # 創建三個欄位 embarked_X
        self.df = self.df.drop("embarked", axis=1) # 刪除原本欄位
        self.df = self.df.join(enbarked_one_hot)   # 加入三個欄位
    def move_target_to_last(self):
        """將目標欄位移至 DataFrame 的最後"""
        df_survived = self.df.pop("survived") 
        self.df["survived"] = df_survived
        print(self.df.head())
    def split_and_save(self, train_csv, test_csv):
        """將資料集分割為訓練集和測試集(80/20%)，並保存為 CSV 文件"""
        mask = np.random.rand(len(self.df)) < 0.8
        df_train = self.df[mask]
        df_test  = self.df[~mask]
        print("Train:", df_train.shape)
        print("Test:",  df_test.shape)
        # 儲存處理後的資料
        df_train.to_csv(train_csv, index=False)
        df_test.to_csv(test_csv, index=False)

def prepare_data():
    titanic = TitanicDataPreparer(seed=7, data_csv="./titanic_data.csv")
    titanic.drop_columns()
    titanic.fill_missing_values()
    titanic.encode_categorical_data()
    titanic.move_target_to_last()
    titanic.describe()
    titanic.split_and_save("titanic_train.csv", "titanic_test.csv")

# 使用範例
if __name__ == "__main__":
    prepare_data()
        
        
