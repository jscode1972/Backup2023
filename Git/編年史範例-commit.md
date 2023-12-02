## git add
加入暫存區
```
$ git status
$ git add *.html
$ git add .
$ git add --all       # 與上面等效
```

## git commit
提交倉庫存檔
```
$ git status
$ git commit -m "說明"
$ git commit --allow-empty -m "空的"     # 可產生空commmit, 方便練習
$ git commit -a -m "跳過add"             # 跳過add,直接commit(僅限於已入庫的檔案)
$ git commit --amend --no-edit          # 不小心漏檔,追加檔案到最近一次 commit
```
