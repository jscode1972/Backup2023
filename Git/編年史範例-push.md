## git push
將本地分支的更改推送到遠端存儲庫
```
# 這行指令只需要輸入第一次即可
git remote add origin <your url>

# 以下兩個指令都會將本地分支與遠端分支關聯起來
$ git push -u origin feature-branch
$ git push --set-upstream origin feature-branch

# 使得後續的 git push 可以直接使用而不需要指定遠端分支和本地分支。
$ git push 
```
