## git remote
加入遠端數據庫 origin 
```
# 這行指令只需要輸入第一次即可 -> 所以在辦公室"開發&掃描"可以分開??
$ git remote add 代表節點 <your url>
$ git remote add origin https://github.com/xxx/zzzz.git

# 修改成 ssh 標記
$ git remote set-url origin
$ git remote -v

# 以下兩個指令都會將本地分支與遠端分支關聯起來
$ git push -u origin feature-branch
$ git push --set-upstream origin feature-branch

# 使得後續的 git push 可以直接使用而不需要指定遠端分支和本地分支。
$ git push

# 若沒設定 upstream, 每次都要講清楚說明白
$ git push 代表節點 分支 
$ git push origin master
$ git push origin master:master  # 等效
$ git push origin master:cat     # 更名遠端
```

## 關於 origin
- origin 僅僅是代名詞, 可自訂替換
- 若是遠端clone下來, 則遠端預設即為 origin
