## 閒聊框架
* numpy => N-dim
* pandas RDB
* scipy scienttific computing
* scikit-learn => machine learning
  * pytorch (臉書)
  * caffe (伯可來)
  * chainer
  * tensorflow (google) keras>
  * https://scikit-learn.org/stable/ 必看
    * Classification
    * Regression
    * Clustering
    * Dimensionality reduction
    * Model Selection
    * ...
    
## 資通電0815 群聊 markdown
* https://bit.ly/3iGy5BP (臉書登入)
  * https://hackmd.io/5uAAezo0So6kwuTqx2YT-w?both

## MAC issue 
* 不支援GPU
* 若裝 nvidia 顯卡, 需要裝 driver

## Windows issue 
* 下載 Visual C++ (MAC 不需要)
  * https://support.microsoft.com/zh-tw/help/2977003/the-latest-supported-visual-c-downloads
    * vc_redist.x64.exe
* 打開長檔名 (256 字元限制)
  * gpedit.msc
    * 系統管理範本/系統/檔案系統
      * 啟用 win32 長路徑 (Win7 沒有, 須上 pack)
        
## prepare env (正常 python)
* `python -m pip install --upgrade pip`
* `pip install -U setuptools`
* `pip install virtualenv virtualenvwrapper virtualenvwrapper-win`
* `mkvirtualenv PYKT_0815`
          
## prepare env (anaconda)
* python -m pip install --upgrade pip
* conda create -n PYKT_0815 python=3.7
  * conda activate PYKT_0815
    * pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-2.1.0-cp37-cp37m-macosx_10_9_x86_64.whl
    * 失敗!
      * pip install tensorflow
  * conda deactivate
  * conda info --envs
* conda env remove --name xyz

## Check
```
python
>>>
import tensorflow as tf
print(tf.__version__)
print(tf.constant("hello tensorflow"))
...
...
tf.Tensor(b'hello tensorflow', shape=(), dtype=string)
```

## install remaining
* pip install numpy scipy sklearn matplotlib pandas ipython jupyter pillow jupyterlab keras
  
## 啟動 PyCharm 
* 新專案 
  * 指定路徑
  * 指定編譯器 PYKT_0815
    * /opt/anaconda3/envs/PYKT_0815/
    * location??
    * interpreter ???
  * 修改 PyCharm 記憶體大小
    * Help / EditCustomVMOption
      * -Xms128m   => 2048m
      * -Xmx1024m  => 2048m

## 早上新專案出問題 (conda)
* 忽略原本指令產出的專案 PYKT_0815
* 重建新專案 xyz
  * 指定專案路徑名稱   /Users/wphuang/Downloads/GitHub/xyz
  * env/xyz         /opt/anaconda3/envs/xyz  (自動產生)
  * 指定python版本   3.7
  * conda 路徑       /opt/anaconda3/bin/conda
* pip install 套件 (進入虛擬環境 xyz)
  * 安裝 tensorflow
  * 安裝 其他 numpy scipy sklearn matplotlib pandas ipython jupyter pillow jupyterlab keras
  * 安裝 kite, cuda
* 此時專案可用 tensorflow
* **回家再做一次一次完成**

## 線上應用
* SymPy https://www.sympy.org/en/index.html
  繪圖練習 * 1/(1+exp(-x))
* https://www.kaggle.com/
* https://www.quantopian.com/
* https://gym.openai.com/
* 賽道 https://aws.amazon.com/tw/deepracer/
* 瑪莉兄弟
* gym.openai.com

## 書籍推薦 (前兩本)
* 深度學習 (Deep Learning) (MIT)
  * https://www.tenlong.com.tw/products/9787115461476?list_name=srh
  * 線上版 https://www.deeplearningbook.org/ (很痛苦, 先看到 9x 頁)
* Foundations of Machine Learning (NYU)
  * https://cs.nyu.edu/~mohri/mlbook/
* https://arxiv.org/abs/1406.2661
* mit Algorithms => Introduction to Algorithms
  * https://www.tenlong.com.tw/products/9780262033848?list_name=srh
* 演算法
  * https://lib.tnml.tn.edu.tw/webpac/search.cfm?m=ss&k0=Algorithms&t0=k&c0=and

# 環境確認
* 程式碼 

# Linear Regression (線性迴歸)
* 以下略, 按筆記上課
* 要好好認識
  * numpy
  * matplotlib.pyplot
  * sklearn
  

          
 
