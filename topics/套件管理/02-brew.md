## 安裝 Homebrew (MAC必備工具, Ruby開發)
以下為我自己(阿賓)親身紀錄  
* 網路好多安裝參考,但是屢裝不好,最後安裝完成卻忘了網址來源
  * 類似: $ruby -e "curl -fssl...."
  * 囧..
* 設定環境變數, 令 /usr/local/bin 在 /usr/bin 之前
  ```$ vim ~/.bash_profile
  export PATH=/usr/local/bin:PATH
  ```
* 指令介紹
  ```
  $ brew list
  $ brew update
  $ brew search 軟件
  $ brew install 軟件
  $ brew update 軟件
  $ brew uninstall 軟件
  ```

## Homebrew 教學
* 參考網址 => [Homebrew 教學 - 以安裝 git 為例](https://w3c.hexschool.com/blog/d2eb4723)
* Mac 預設是沒有安裝 Homebrew 的，所以你必須先透過指令安裝，步驟如下。
  * 打開終端機，輸入以下指令
  > `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

## 安裝套件
安裝套件的方式很簡單，例如安裝 Wget 一樣透過指令：
```
$ brew install wget
```

## 更新套件
一氣呵成的指令內容分別是： 
  * brew update：更新 Homebrew 及套件清單 
  * brew upgrade：更新所有套件 
  * brew cleanup：清除暫存檔   

不過在 brew 軟體清單內可能大部分都是比較「工程」的套件，因此如果要安裝一些「尋常」的軟體，
我們可以安裝 Homebrew-Cask 這套軟體，來取得更多的應用程式。
```
$ brew update && brew upgrade && brew cleanup
Error:  /usr/local must be writable!
```
