## 專案入口網址
* https://console.cloud.google.com/
* https://console.firebase.google.com/ (Firebase 專案也就是 Cloud 專案。)

## 控制台建立專案 ( Firebase Console)
建立專案 GTD-2024 

## 升級 npm 
升級 npm (須透過 n 更新)
```
$ npm install -g n
$ sudo n lts      (node/npm 長期版本) 或
$ sudo n latest   (node/npm 最新版本)
```
安裝 firebase-tools SDK
```
$ npm install -g firebase-tools (@13.0.2)
```

## 全新空白專案 firebase init (Hosting)
建立 hosting (須重裝 firebase-tools, 順帶升級 node/n/npm )
```
# 準備網頁應用 (後面 init 需要指定專案 index.html)
# public/index.html
# src/index.html

# 初始化 firebase
$ firebase login  (準備登入 google)
$ firebase login:ci
$ firebase login --reauth (重新驗證, 驗證後才能安裝套件?)
$ firebase projects:list  (列出專案)
$ firebase init           (啟動專案-法1, 選取 hosting)
  ❯◉ Hosting: Set up GitHub Action deploys (一直失敗)
$ firebase init hosting   (啟動專案-法2, 帶參數 hosting)
  ? Please select an option: Use an existing project => 指定現成專案
  ? Select a default Firebase project for this directory: => 輸入專案名稱 gtd-2024
  ? What do you want to use as your public directory? (public)  => 指定輸出位置 dist/gtd-2024 (Angular)
  ? Configure as a single-page app (rewrite all urls to /index.html)? (y/N) => 不覆蓋
  ? Set up automatic builds and deploys with GitHub? (y/N) => N

# 此時產生設定檔
- firebase/
- .firebaserc
- firebase.json

# 部署站台
$ firebase deploy  (部署, 可將網頁上傳, 有兩種網域)
```

## 現成 Angular 專案 (項目選擇)
準備 /src/environments/environment.ts ??? 殺小?
```
# 重新驗證, 驗證後才能安裝套件
$ firebase login --reauth
# 安裝套件及項目
$ ng add @angular/fire
```
操作範例
```
? What features would you like to setup? (Press <space> to select, <a> to toggle all, <i> to invert selection, and <enter> to proceed)
? What features would you like to setup? ng deploy -- hosting, Firestore, Analytics
 ◉ ng deploy -- hosting
 ◯ Authentication
 ◉ Firestore
❯◯ Realtime Database
 ◉ Analytics
 ◯ Cloud Functions (callable)
 ◯ Cloud Messaging

? Which Firebase account would you like to use? (Use arrow keys)
  [Login in with another account] 
❯ xxxxxx@gmail.com

Already using account xxxxx@gmail.com for this project directory.

```

