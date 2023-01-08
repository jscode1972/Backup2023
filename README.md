## Backup2023
存放一些暫存東西 (過期的,準備整理的,新想法..)

## git 指令
```
$ git clone https://xxxx/xxxx.git
$ rm -rf .git

# 檔案列管, 開始追蹤一舉一動 (馬上產生 objects/xx/hash檔), 此時 graph/hstory 還抓不到資料
$ git add .                 # 僅有此目錄,及子目錄,孫目錄..(其他目錄不管)
$ git add --all             # 專案所有目錄
$ git reset                 # 反悔, 取消上述動作  => 比較 $ git reset HEAD ??  
$ git commit -m "說明..."    # 提交成果到檔案庫
$ git add . &&  git commit -m "說明..."
$ git rm 檔案                # deleted: 刪檔案且自動add狀態改變
$ git rm 檔案 --cached       # untracked: 脫離git控管 (非真正刪除檔案) 
$ git mv 檔名A 檔名B          # renamed: 改檔名 
----------------------------------------------
$ git log --oneline --graph              # 圖形顯示
$ git log --author="甲\|乙" --grep="wtf"  # 過濾作者/過濾commit字串
$ git log -S "內容"                       # 找出內容含有..字串
$ git log --since="9am" util="12am"      # 找出時間
$ git log --after="2023-01"              # 找出日期
----------------------------------------------
$ git cat-file -t hash     # 查看什麼型態
$ git cat-file -p hash     # 查看什麼內容
----------------------------------------------
$ git push                 # 上傳
$ git push -u master

# 如果衝突,先把上面的東西拉下來
$ git pull
跳出內容,編輯 :wq

```

## git 
google: git pull fetch  
 * [Pull 下載更新 (不錯的網頁說明清楚, 但是 upstream -u 是什麼呢??)](https://gitbook.tw/chapters/github/pull-from-github)
 * [What difference between 'pull' & 'fetch'?](https://stackoverflow.com/questions/292357/what-is-the-difference-between-git-pull-and-git-fetch) (很多圖片)
   * xxx

## .git/ 裡面是啥摸 hash？


# 測試指令
```
SHA-1    說明          cat  dev  rels  master/dog
-------- ------------ --------------------------
018a659d 初次commit                                              
-------- ------------ --------------------------                                  
                                                                                                                                              
                                                                                                                                                                                                                          

```
