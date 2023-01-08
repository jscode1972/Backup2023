## GCP 建立專案
* [專案平台](https://console.cloud.google.com/)
* [建立+部署範例](https://codelabs.developers.google.com/codelabs/cloud-app-engine-python3/#0) 照做就會成功!!
* 本節重點 (本地指令 vs 雲端介面)
  * [本地指令建立專案](#本地指令建立專案)
  * [本地建立程式專案](#本地建立程式專案) (本節另開章節)
    * PyCharm 開啟新專案
    * 設定虛擬環境 venv 解譯器
    * 加入專案檔 (app.yaml, requirements.txt, main.py)
  * [本機部署專案到雲端](#本機部署專案到雲端)
  
### 本地指令建立專案
```
## 建立專案
$ gcloud projects create 小寫專案ID --name=專案名稱
$ gcloud projects create gcp-20200913 --name=GCP-20200913
$ gcloud projects create cmd-20200913 --name=CMD-20200913

## 查看帳號是否已授權 
$ gcloud auth list
  Credentialed Accounts
    ACTIVE  ACCOUNT
      wphuang@gmail.com

## 若否, 切換帳號 or 執行授權 SDK
$ gcloud config set account `ACCOUNT`
$ gcloud auth login 

## 查看目前是否為目標專案
$ gcloud config list project
  [core]
    project = gcp-20200913        (這個其實雲端已經刪除了)

## 若否, 更換專案
$ gcloud config set project <PROJECT_ID>
$ gcloud config set project cmd-20200913
  Updated property [core/project].
```

### 表格圖示
| 指令                   | API金鑰 | OAuth2 用戶端ID | 服務帳戶                   |  資源  | 帳單帳戶          | API 程式庫 |
|:------------------------|:--------:|:----------:|:--------------------------:|:-----:|:---------------:|:----------:|
|gcloud projects create | 無      | 無             | 無(憑證/IAM 皆無)           | 無    | 無                | 無        |
|啟用 Cloud Build API    | 無      | 無             | cmd-20200913@appspot.gs...| 無    | 無                | 無        |
|啟用計費功能             | 無      | 無             | 無                         | 無    | 帳單帳戶-20200913  | 無        |

* [備註]  cmd-20200913@appspot.gserviceaccount.com

### 本地建立程式專案 (Python => 下一節)
#### 快速帶過, 請看
```
* PyCharm 開啟新專案
* 設定虛擬環境解譯器 (venv)
* 安裝套件 flask 
* 加入專案檔 
  * app.yaml
  * requirements.txt
  * main.py
```

### 本機部署專案到雲端
#### 步驟說明
* 執行部署指令 => gcloud app deploy
* 選擇區域 => asia-east2
* 確認訊息 => y
* 完成上傳 => 出現錯誤訊息 (未開啟 Cloud Build API 功能)
  * https://console.developers.google.com/apis/api/cloudbuild.googleapis.com/overview?project=cmd-20200913
  * 啟用 Cloud Build API 功能
  * 必須啟用計費功能 => 設定帳單帳戶
* 打開連結, 參考下一節做法

#### 執行部署指令
```
## 執行部署指令
$ gcloud app deploy

## 選擇區域 
Please choose the region where you want your App Engine application
located:
 [1] asia-east2    (supports standard and flexible)
Please enter your numeric choice:  1        (選日本)

## 確認訊息
Creating App Engine application in project [cmd-20200913] and region [asia-east2]....done.                                                                
Services to deploy:

descriptor:      [/Users/wphuang/Github/cmd-20200913/app.yaml]
source:          [/Users/wphuang/Github/cmd-20200913]
target project:  [cmd-20200913]
target service:  [default]
target version:  [20200914t010030]
target url:      [https://cmd-20200913.df.r.appspot.com]

## 完成上傳
Beginning deployment of service [default]...
Created .gcloudignore file. See `gcloud topic gcloudignore` for details.
╔════════════════════════════════════════════════════════════╗
╠═ Uploading 8 files to Google Cloud Storage                ═╣
╚════════════════════════════════════════════════════════════╝
File upload done.

## 出現錯誤訊息
Updating service [default]...failed.                                                                                                                      
ERROR: (gcloud.app.deploy) Error Response: [7] Access Not Configured.
Cloud Build has not been used in project cmd-20200913 before or it is disabled. 
Enable it by visiting https://console.developers.google.com/apis/api/cloudbuild.googleapis.com/overview?project=cmd-20200913 then retry.
If you enabled this API recently, wait a few minutes for the action to propagate to our systems and retry.
```

#### 打開連結
* https://console.developers.google.com/apis/api/cloudbuild.googleapis.com/overview?project=gcp-20200913
* 前步驟部署失敗, 需啟用Cloud Build API
  * 還要設定帳單帳戶
  * 必須啟用計費功能 => 設定帳單帳戶
  * 您必須具備憑證，才能使用這個 API。首先，請點選 [建立憑證]。
    [ ] API 金鑰
    [ ] OAuth 2.0 用戶端 ID
    [x] 服務帳戶 (gcp-20200913@appspot.gserviceaccount.com)
* 再次部署 (gcloud app deploy)
  * https://gcp-20200913.appspot.com/
  
```
descriptor:      [/Users/wphuang/Github/cmd-20200913/app.yaml]
source:          [/Users/wphuang/Github/cmd-20200913]
target project:  [cmd-20200913]
target service:  [default]
target version:  [20200914t012035]
target url:      [https://cmd-20200913.df.r.appspot.com]
```
