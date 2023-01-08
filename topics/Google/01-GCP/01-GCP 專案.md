# GCP 專案建立
* [專案平台](https://console.cloud.google.com/)
  * https://console.cloud.google.com/

## 建立專案 
#### 建立專案
* 1.雲端介面建立專案 (設定專案名稱 & ID)
  * 專案名稱  "GCP-20200913"
  * 專案ID   "gcp-20200913" (小寫)
* 2.本地指令建立專案
  * gcloud projects create 小寫專案ID --name=專案名稱
    * gcloud projects create gcp-20200913 --name=GCP-20200913
#### 查看帳戶指令
  * gcloud auth list
  * gcloud auth login  (切換登入帳號)
#### 查看專案指令
  * gcloud config list project
#### 更換專案指令
  * gcloud config set project gcp-20200913

    
