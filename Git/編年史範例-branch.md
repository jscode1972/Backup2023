## git branch
建立分支 (分支就是一張貼紙,一個指標,貼在commit上而已)
```
$ git branch              # 列出分支
$ git branch cat          # 建立分支
$ git branch -m cat dog   # 分支改名
$ git branch -d cat       # 刪除分支 (沒合併無法)
$ git branch -D cat       # 刪除分支 (強迫刪除, 可救回來)
```

切換分支
```
$ git checkout cat        # 切換分支
$ git checkout -b cat     # 切換分支 (強迫建立)
```
