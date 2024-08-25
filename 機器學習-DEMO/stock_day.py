""" 臺灣證券交易所-OpenAPI
這個範例示範了如何呼叫 [臺灣證券交易所-OpenAPI] 
Todo:
    * For module TODOs
---------------------------------------------------------------------------------------------
@Author: 黃小兵
@Date:   2024/08/25
@Links:  https://www.twse.com.tw/ (官網)
@refer:  https://stackoverflow.com/questions/1523427/what-is-the-common-header-format-of-python-files
"""
import json
import pandas as pd
import requests
import glob
import os
import csv
import time

class APIClient:
    def __init__(self, base_url, output_dir):
        self.base_url = base_url
        self.output_dir = output_dir

    def fetch_data(self, endpoint, params=None):
        """ 通用的 API 呼叫方法，發送 GET 請求並返回 JSON 回應
        :param endpoint: API 的端點
        :param params: API 請求的參數 (字典格式)
        :return: JSON 格式的資料
        """
        response = requests.get(self.base_url + endpoint, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API 請求失敗，狀態碼: {response.status_code}")

    def save_to_csv(self, data, output_csv):
        """ 將資料存為 CSV 檔案
        :param data: 要存的資料 (通常是 JSON)
        :param output_csv: 輸出的 CSV 檔案名稱
        """
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"資料已存為 '{output_csv}'")

class StockDayAllAPI(APIClient):
    # https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL
    def __init__(self, base_url, output_dir):
        super().__init__(base_url, output_dir) 

    def fetch_and_save(self, prefix='stock_day_all'):
        """ 呼叫 STOCK_DAY_ALL API 並將結果存為 CSV 檔案
        :param prefix: 輸出的 CSV 檔案名稱的前綴
        """
        endpoint = '/exchangeReport/STOCK_DAY_ALL'
        data = self.fetch_data(endpoint)
        output_csv = f"{self.base_dir}/{prefix}.csv"
        self.save_to_csv(data, output_csv)

class StockDayPerAPI(APIClient):
    # https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20240825&stockNo=2330&response=json&_=1724570424858
    def __init__(self, base_url, output_dir):
        super().__init__(base_url, output_dir) 

    def fetch_and_save(self, stockNo, date, prefix='stock_day'):
        """ 呼叫 Example API 並將結果存為 CSV 檔案
        :param param1: 第一個參數
        :param param2: 第二個參數
        :param prefix: 輸出的 CSV 檔案名稱的前綴
        """
        endpoint = '/rwd/zh/afterTrading/STOCK_DAY'
        # 生成當前時間戳
        timestamp = int(time.time() * 1000)  # 轉換為毫秒
        params = {'date': date, 'stockNo': stockNo, 'response': 'json', '_': timestamp }
        data = self.fetch_data(endpoint, params)
        # 將 JSON 解析為字典
        #data = json.loads(json_data)
        # 提取標題和數據
        fields = data["fields"]
        rows = data["data"]
        # 去掉 rows 中的數字中的逗號
        cleaned_rows = []
        for row in rows:
            cleaned_row = [col.replace(',', '') if col.replace(',', '').isdigit() else col for col in row]
            cleaned_rows.append(cleaned_row)
        # 解析日期，生成年月
        yyyy = date[:4]  # YYYY 格式
        yyyymm = date[:6]  # YYYYMM 格式
        output_csv = os.path.join(self.output_dir, stockNo, f"{prefix}_{stockNo}_{yyyymm}.csv")  
        print(output_csv)
        #self.save_to_csv(data, output_csv)
        # 將數據寫入 CSV 文件
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 寫入標題
            writer.writerow(fields)
            # 寫入數據行
            writer.writerows(cleaned_rows)
    
    def merge_monthly_csv_files(self, stockNo, year, prefix='stock_day'):
        """合併特定年份的多個月度 CSV 文件成一個年度 CSV 文件。
        :param base_dir: CSV 文件所在的目錄
        :param stock_no: 股票代碼
        :param year: 要處理的年份
        :param prefix: 輸出的 CSV 檔案名稱的前綴
        """
        # 查找所有符合條件的文件
        file_pattern = os.path.join(self.output_dir, stockNo, f"{prefix}_{stockNo}_{year}[0-1][0-9].csv")
        files = glob.glob(file_pattern)
        
        # 確保按照月份順序進行排序
        files.sort()  # glob.glob 的結果會按字母順序返回，通常符合月份順序，但這樣更保險
        
        # 定義輸出文件的名稱
        output_csv = os.path.join(self.output_dir, stockNo, f"{prefix}_{stockNo}_{year}.csv")
        # 如果目標文件已存在，則刪除它
        #f os.path.exists(output_csv):
        #    os.remove(output_csv)
            
        # 初始化存儲數據的列表
        all_rows = []
    
        for file in files:
            with open(file, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # 讀取標題行
                if not all_rows:  # 如果是第一次讀取，保存標題行
                    all_rows.append(header)
                all_rows.extend(list(reader))  # 添加所有數據行
                """ # 去掉千位逗號 (來源已經清除千進位,所以不需要)
                for row in reader: 
                    row = [col.replace(',', '') if col.replace(',', '').isdigit() else col for col in row]
                    all_rows.append(row)
                """
        # 將合併後的數據寫入新的 CSV 文件
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(all_rows)

# 呼叫 STOCK_DAY API 並存檔
TWSE_BASEURL = 'https://www.twse.com.tw'
stock = '2330'
date  = '20240101'
output_dir = os.path.expanduser('~/PYTHON-TWSE-API/TWSE-data/stock_day')
stock_day_api = StockDayPerAPI(TWSE_BASEURL, output_dir)
stock_day_api.fetch_and_save(stock, date)
stock_day_api.merge_monthly_csv_files(stock, date[:4])
