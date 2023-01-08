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
------------------------------------------------------------------
# 以下動作儘量不要用在已經 push 的 commit
$ git commit --amend -m "修改commit內容註解"  # 修改前一次commit註解   
$ git commit --amend --no-edit              # 追加檔案不想再一次 commit (法2,直接reset,重新commit)
------------------------------------------------------------------
# add .gitignore (ex: 忽略副檔名 *.xxx )   
$ git rm 檔名 --cached                    # 將本該忽略的傢伙請出去 (移出控管)
$ git add -f filename.xxx                # 忽略這次忽略(開例外)
$ git clean -fx                          # 刪掉被忽略的檔案
-------------------------------------------------------------------
$ git reflog                 # 刪錯想回頭,沒記住之前的版次代碼,重新叫出來
$ git blame [-L 起,迄] 檔名   # 查看某一行是誰加進去的 (-L 指定行數區間)
-------------------------------------------------------------------
$ git checkout 檔名 (.)      # 救回不小心刪掉的檔案(deleted), (checkout亦可針對分支) 
$ git checkout HEAD~2 檔名   # 將前兩個版本前的檔案覆蓋回來 (同時更新狀態)
-------------------------------------------------------------------
$ git reset e12d81s         # (絕對) 回到過去某版本 (指定版次)
$ git reset e12d81s^^       # (相對) 回到過去某版本 (有n個^ 就往前推n個版本)
$ git reset master^^        # (相對) 回到過去某版本 (HEAD亦可)
$ git reset HEAD~5          # (相對) 回到過去某版本 (太多可標示~n)
$ git reset 分支 [--mixed]   # 丟回工作目錄
$ git reset 分支 --soft      # 丟回暫存區
$ git reset 分支 --hard      # 刪掉檔案
$ git reset 未來分支          # 萬一倒退嚕過頭,想要回頭 (--hard, 強迫放棄之後修改的檔案)
-------------------------------------------------------------------
$ git log --oneline --graph              # 圖形顯示
$ git log --author="甲\|乙" --grep="wtf"  # 過濾作者/過濾commit字串
$ git log -S "內容"                       # 找出內容含有..字串
$ git log --since="9am" util="12am"      # 找出時間
$ git log --after="2023-01"              # 找出日期
-------------------------------------------------------------------
$ git log 檔名               # 查看指定檔案的 commit 紀錄
$ git log -p 檔名            # 查看指定檔案的 commit 紀錄 & 修改
-------------------------------------------------------------------
```

# git 物件架構 (重要觀念 ref 高見龍 p.109)
* 四大天王 (Commit->Tree->Blob & Tag)
* branch 指向某個 commit (branch/remote)
* HEAD 指向某個分支
```
# 建立/切換分支 (高見龍 p.130~132, 圖示清楚)
$ git branch [分支]            # 查看分支狀態/建立
$ git branch -m old new       # 更名分支
$ git branch -d name          # 刪除分支 (沒有分支不能刪的,但是要先切換)
$ git branch -D name          # 刪除分支 (尚未合併不可刪 => 改大寫D, 強迫刪除)
$ git checkout 分支            # 切換分支 (比較: 前面 checkout filename)
$ git branch new_name b14a1b3 # 不小心砍掉分支,弄一個新標籤黏回去 (commit 還在)
-------------------------------------------------------------------
# 合併分支 (法1, master 併 develop)  (高見龍 p.137)
$ git checkout master
$ git merge develop
-------------------------------------------------------------------
# 合併三支 (法2, 高見龍 p.154)
$ git rebase xxx  待研究 待研究 待研究
$    
-------------------------------------------------------------------
# 合併衝突
$                  待研究 待研究
-------------------------------------------------------------------
$ git gc                     # 資源回收
$ find .git/objects -type f  # 查看
$ git verify-pack -v .git/objects/pack/pack-hash碼.pack   # 查看進一步資訊
# 查看內容
$ git cat-file -t hash      # 查看什麼型態
$ git cat-file -p hash      # 查看什麼內容
-------------------------------------------------------------------
# 上傳
$ git push                
$ git push -u master
-------------------------------------------------------------------
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
