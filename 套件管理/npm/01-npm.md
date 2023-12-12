## 常用 npm 指令
* [套件命名規則](#套件命名規則)
* [list](#list) 列出已安裝套件
* [install-g](#install-g) 全域安裝/移除
* [install](#install) 專案安裝/移除
* [update](#update) 更新套件
* [info](#info) 套件資訊
* [search](#search) 搜尋套件
* [package.json](package.json) 管理套件

--------------------------------
## 套件命名規則
```
* 套件名稱
- npm install (with no args, in package dir)
- npm install [<@scope>/]<name>
- npm install [<@scope>/]<name>@<tag>
- npm install [<@scope>/]<name>@<version>
- npm install [<@scope>/]<name>@<version range>
- npm install <git-host>:<git-user>/<repo-name>
- npm install <git repo url>
- npm install <tarball file>
- npm install <tarball url>
- npm install <folder>
```

--------------------------------
## list
可以查看**安裝的路徑**，以及相依，以階層顯示。
```
$ npm ls -g              # 列出全域裡的套件
$ npm ls -g -l 套件名稱   # 列出全域裡的套件詳細資訊
$ npm ls 套件名稱         # 列出專案裡的套件
$ npm ls -l 套件名稱      # 列出專案裡的套件詳細資訊
$ npm ls --depth=0       #只顯示第一層
```
--------------------------------
## install-g
**全域安裝**的套件通常只是為了**執行檔**而已
```
$ npm install -g {套件名稱} 
# 範例 
$ npm install -g express               # 安裝模組,需手動在 package.json 加入依賴)
$ npm install -g express --save        # 安裝模組,連同版次寫入 dependencies 正式)
$ npm install -g express --save--dev   # 安裝模組,連同版次寫入 dependencies 開發)
------------------------------------------------------------------------------
-P, --save:          Package will appear in your dependencies.
-D, --save-dev:      Package will appear in your devDependencies.
-O, --save-optional: Package will appear in your optionalDependencies.
--no-save:           Prevents saving to dependencies.
------------------------------------------------------------------------------
$ express new app                      # 安裝完後現在我們可以用 express 來產生專案
------------------------------------------------------------------------------
$ npm uninstall -g {套件名稱}
$ npm uninstall -g express
```
--------------------------------
## install
將套件安裝在**專案**裡. 每一個不同的專案裡都要重裝一次套件. 不然會 require 不到.
```
$ cd /path/to/the/project
$ npm install {套件名稱}
$ npm install express      # 現在就可以在專案裡用 var express = require( 'express' ); 來使用 express 這個套件了.
```
--------------------------------
## update
**更新全域/專案套件** 不加 -g 就是當前專案 
```
$ npm update -g (更新全域裡的套件)
$ npm update    (更新專案裡的套件)
```
--------------------------------
## info
**套件資訊** 查看套件詳細內容(官網/維護人員/當前版本/未來版次)
```
$ npm info {套件名稱} <屬性>
$ npm info express version (查看版本)
$ npm info express         (顯示更多)
```
--------------------------------
## search 
**搜尋套件** 顯示有此_關鍵字_的套件 (以表格顯示)
```
$ npm search {套件名稱}
$ npm search express
```
--------------------------------
## package.json 
只要將 package.json 這個檔案放在專案的根目錄裡, 就不需要一個個的手動安裝套件.
```
# 範例
{
  "name": "tour-of-heros",
  "version": "0.0.0",
  "scripts": {
    "ng": "ng",
    "start": "ng serve",
    "build": "ng build",
    "test": "ng test",
    "lint": "ng lint",
    "e2e": "ng e2e"
  },
  "private": true,
  "dependencies": {
    "@angular/animations": "~7.1.0",
    "@angular/common": "~7.1.0",
    ....
    "core-js": "^2.5.4",
    "rxjs": "~6.3.3",
    "tslib": "^1.9.0",
    "zone.js": "~0.8.26"
  },
  "devDependencies": {
```
