## 遠端登入 Github
創建新專案 test
  * 設定: 名稱/說明/private 
  * 此時回傳專案網址
    * https://github.com/jscode1972/test.git

## 綁定本地專案 (command line)
若尚未登錄過, 會要求輸入 github 帳密
#### <法一> 創建全新專案 (create a new repository on the command line)
```
$ echo "# test" >> README.md
$ git init
$ git add README.md
$ git commit -m "first commit"
$ git remote add origin https://github.com/jscode1972/test.git
$ git push -u origin master
```
#### <法二> 綁定現有專案 (push an existing repository from the command line)
```
$ git remote add origin https://github.com/jscode1972/test.git
$ git push -u origin master 
```

## 複製他人專案
記得先看 README.md 說明檔，參考大漠窮秋@github (NiceFish & Angular)
```
$ git clone https://github/帳號/專案.git
```
