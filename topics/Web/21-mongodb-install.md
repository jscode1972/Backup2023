#### 安裝 MongoDB 伺服器 (先跳過)
$ brew insatll mongodb (MAC)

#### 免費 MongoDB 伺服器 (雲端)
* 官網註冊 js/大名(5)年(4)金聖嘆
* 選擇 cluster / 免費地區 / 維吉尼雅  
* 創建新帳號 1193/za-lm(8)
* 設定白名單 host/ip (老家/新竹/免費wifi)
* 測試連線 Studio-3T 
  + 管理介面 (Cluster)
    - 按下 Connect (會出現連線字串, copy)
    - 打開連線工具

#### 安裝連線工具 Studio-3T (Mac)
+ 到 robomongo 的官網下載應用程式
* 測試連線,打開 Studio-3T
  + 貼上 from URL (敲密碼
  + 連不上有可能雲端白名單 ip 要重設
  + OK

#### 安裝 node client
$ npm install -g mongodb
$ mkdir mongo-test
$ cd mongo-test
$ npm init
$ npm install mongodb --save
$ 貼上程式碼
$ 一堆網頁語法過時 (自己踹)
$ 語法說明 => https://docs.mongodb.com/manual/reference/method/
  

參考文件  
* 官網 (NodeJS)
  + https://docs.mongodb.com/manual/tutorial/getting-started/
* 官網 https://docs.mongodb.com/manual/reference/method/
* 官網 http://mongodb.github.io/node-mongodb-native/3.1/quick-start/quick-start/
