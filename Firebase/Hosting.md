# 學習重點
- 控制台建立專案
- 升級 npm / n
- 安裝 firebase SDK ( firebase-tools )
- 綁定/部署 hosting
  - firebase login
  - firebase init
  - firebase deploy
- 進階套件 ( @angular/fire )
  - ng add @angular/fire

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

## 安裝 firebase SDK 
安裝 firebase-tools 
```
$ npm install -g firebase-tools (@13.0.2)
```

## 綁定 hosting (網站代管)

### 建立專案
準備網頁應用 (後面需要指定專案入口 index.html)
- 空白專案
  - firebase init (自動產生有 library index.html)
- Angular

### 須安裝/重裝 firebase-tools
- 安裝 firebase-tools (參考前面)
- 順帶升級 node/n/npm (參考前面)

### 登入 firebase
$ firebase login  (登入)
```
$ firebase login  (準備登入 google)
$ firebase login:ci
$ firebase login --reauth (重新驗證, 驗證後才能安裝套件?)
$ firebase projects:list  (列出專案)
```

### 空白專案 (初始/部署)
初始 firebase, 自動產生有 library 的 index.html
```
$ firebase init [hosting]
# 選擇初始項目
? Which Firebase features do you want to set up for this directory?
Press Space to select features, then Enter to confirm your choices. 
❯◉ **Hosting: Configure files for Firebase Hosting and (optionally) set up GitHub Action deploys**
# 選擇專案 (firebase)
? Please select an option: (Use arrow keys)
❯ **Use an existing project**
❯ gtd-2024 (GTD-2024)
# 指定目錄
? What do you want to use as your public directory? (public)
# 指定入口網頁
? Configure as a single-page app (rewrite all urls to /index.html)? (y/N)

# 產生設定檔
firebase.json

# 部署站台
$ firebase deploy  (部署, 可將網頁上傳, 有兩種網域)
Project Console: https://console.firebase.google.com/project/gtd-2024/overview
Hosting URL: https://gtd-2024.web.app
```

### Angular 專案限定 (初始/部署)
額外安裝套件及項目???
```
# 安裝 angular
$ ng new 專案

$ ng add @angular/fire
? What features would you like to setup? ng deploy -- hosting

# 一直報錯!!!!firebase-tools@13
# 降級 ok
$ npm i -g firebase-tools@12.9.1
# 但是還是有問題

```

