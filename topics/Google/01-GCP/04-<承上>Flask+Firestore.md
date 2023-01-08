參考 https://firebase.google.com/docs/admin/setup?authuser=0#python    
搭配  https://console.firebase.google.com/project/test-20200816/settings/serviceaccounts/adminsdk

## 編修 Flask 專案
#### 安裝套件 firebase_admin, 需先進入虛擬模式 
* 打開專案  => 自動進入虛擬模式
  * 或是指令進入 => 虛擬模式
  `$ conda activate cmd-20200913`
```
## 安裝套件 firebase_admin
(cmd-20200913) $ conda install firebase_admin
## 查看 firebase_admin 版次 1.1.2
(cmd-20200913) $ conda list firebase-admin       # 為什麼前面是底線, 安裝完是 dash??
```

## 下載 API 憑證 (credential)
#### test-20200816.json (憑證)
cmd-20200913 沒有 firesotre => 暫時用 test-20200816

### 編修程式專案 [參考網址](https://medium.com/pyradise/10%E5%88%86%E9%90%98%E8%B3%87%E6%96%99%E5%BA%AB%E6%93%8D%E4%BD%9C-%E6%96%B0%E5%A2%9E%E8%B3%87%E6%96%99-b96db385e1e4)
* app.yaml (不動)
* requirements.txt (用了什麼套件就要加上)
* test-20200816.json (憑證)
* main.py

### 範例檔
#### app.yaml (不動)
```
runtime: python38
```
#### requirements.txt (用了什麼套件就要加上)
```
firebase-admin==4.3.0
Flask==1.1.2
```
#### test-20200816.json
```
{
  "type": "service_account",
  "project_id": "test-20200816",
  "private_key_id": "............",
  "private_key": "-----BEGIN PRIVATE KEY----- ....",
  "client_email": "firebase-adminsdk-p0jyz@test-20200816.iam.gserviceaccount.com",
  "client_id": ".......",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-p0jyz%40test-20200816.iam.gserviceaccount.com"
}
```
#### main.py
```python
from flask import Flask
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

##
cred = credentials.Certificate("./test-20200816.json")
firebase_admin.initialize_app(cred)

def xx():
  ## https://medium.com/pyradise/10%E5%88%86%E9%90%98%E8%B3%87%E6%96%99%E5%BA%AB%E6%93%8D%E4%BD%9C-%E6%96%B0%E5%A2%9E%E8%B3%87%E6%96%99-b96db385e1e4
  # 初始化firestore
  db = firestore.client()
  doc = {
    'name': "帽子哥",
    'email': "abc@gmail.com"
  }
  # 建立文件 必須給定 集合名稱 文件id
  # 即使 集合一開始不存在 都可以直接使用
  # 語法
  # doc_ref = db.collection("集合名稱").document("文件id")
  doc_ref = db.collection("students").document("student_03")
  # doc_ref提供一個set的方法，input必須是dictionary
  doc_ref.set(doc)

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    xx()
    """Return a friendly HTTP greeting."""
    return 'Hello World! 成功3!\n'

if __name__ == '__main__':
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='localhost', port=8080, debug=True)
```
