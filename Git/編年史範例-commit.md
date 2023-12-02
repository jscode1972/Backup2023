## git add
加入暫存區
```
$ git status
$ git add *.html
$ git add .
$ git add --all       # 與上面等效
```

## .gitignore
例外檔案
```
$ git add -f 檔案      # 強迫加入檔案 (避開 .gitignore 限制)
$ git rm --cached     # 移出 .gitignore 所列管的檔案
$ git clean -fx       # 刪除已被忽略的檔案 (實體檔案,若已被追蹤的不刪)
```

## git commit
提交倉庫存檔
```
$ git status
$ git commit -m "說明"
$ git commit --allow-empty -m "空的"     # 可產生空commmit, 方便練習
$ git commit -a -m "跳過add"             # 跳過add,直接commit(僅限於已入庫的檔案)
```

## git commit --amend
修改記錄 (--amend 只能修改最後一次), 儘量不要使用在push出去的commit
```
$ git commit --amend --no-edit          # 不小心漏檔,追加檔案到最近一次 commit
$ git commit --amend -m "修改註解"        # 不小心打錯,修改最近一次 commit 註解 (也會追加檔案)
```
