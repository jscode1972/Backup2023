## 建立Python專案 (本機)
#### 本節重點
* [專案平台](https://console.cloud.google.com/)
* [建立程式專案](#創建程式專案) => PyCharm
  * 建立程式專案 (預設 main.py)
  * 建立虛擬環境
* [安裝套件](#安裝套件)
  * conda install flask
* [編修程式專案](#編修程式專案) => Python
  * 安裝套件 flask
  * 加入 app.yaml
  * 加入 requirements.txt
  * 編修 main.py 

### 創建程式專案
####  創建程式專案,並創建虛擬環境
* 啟動 PyCharm
* New Project:
  * **cmd-20200913**
* Location (指定專案路徑)
  * /Users/wphuang/GitHub/**cmd-20200913**  (此目錄不用預先創建)
* Python Interpreter (指定直譯器)
  - [x] **New Enviroment** using <conda>
    * Location: /opt/anaconda3/envs/**cmd-20200913** (自動跟著上面,不要改,不用預先創建虛擬環境)
    * Python Version: 3.8
    * Conda Executable: /opt/anaconda3/bin/conda
  - [ ] Existing Interpreter
    * Interpreter
      - [ ] Virtualenv Enviroment
      - [ ] Conda Enviroment 
      - [ ] System Interpreter 
      - [ ] Pipe Enviroment
* 注意: envs 實際路徑視anaconda原始安裝路徑而定
  * Anaconda 不一定裝在 /opt (可能是用了 sudo)

### 安裝套件
#### 安裝套件 flask, 需先進入虛擬模式 
```
## 打開專案 => 自動進入虛擬模式
## 或是指令進入 
$ conda activate cmd-20200913

## 安裝套件
(cmd-20200913) $ conda install flask

## 查看 falsk 版次 1.1.2
(cmd-20200913) $ conda list flask
```

### 編修程式專案
* [官方範例](https://codelabs.developers.google.com/codelabs/cloud-app-engine-python3/#0) 照做就會成功!!
* 開啟專案 cmd-20200913
* 編修三個檔案分別是
  * 新增 app.yaml
  * 新增 requirements.txt 
  * 編修 main.py
* 程式範例
  * **app.yaml**
    ```
    runtime: python38         # python 版次
    ```
  * **requirements.txt**
    ```
    Flask==1.1.2              # Flask 版次
    ```
  * **main.py** 程式碼
    ```python
    # [START gae_python38_app]
    from flask import Flask
    # If `entrypoint` is not defined in app.yaml, App Engine will look for an app
    # called `app` in `main.py`.
    app = Flask(__name__)
    @app.route('/')
    def hello():
        """Return a friendly HTTP greeting."""
        return 'Hello World!'
    if __name__ == '__main__':
        # This is used when running locally only. When deploying to Google App
        # Engine, a webserver process such as Gunicorn will serve the app. This
        # can be configured by adding an `entrypoint` to app.yaml.
        app.run(host='localhost', port=8080, debug=True)
    # [END gae_python38_app]
    ```

### 執行程式專案
#### PyChrom 右鍵執行
  ```
  * Serving Flask app "main" (lazy loading)
  * Environment: production
     WARNING: This is a development server. Do not use it in a production deployment.
     Use a production WSGI server instead.
  * Debug mode: on
  * Running on http://localhost:8080/ (Press CTRL+C to quit)
  ```

### 手動創建虛擬環境venv
* $ conda create -n 虛擬環境 python=3.7 packages..
* $ conda info --envs (查看所有安裝的虛擬環境)
* $ conda list env (同上查看所有安裝的虛擬環境)
* $ conda list -n 虛擬環境 (查看虛擬環境套件)
* $ conda activate 虛擬環境 (啟用虛擬環境)
  * (虛擬環境) $ conds list (在環境中則不需指定 -n xxx)
  * (虛擬環境) $ conda deactivate (已在虛擬環境, 準備退出)
* $ conda env remove --name 虛擬環境 (移除虛擬環境)
