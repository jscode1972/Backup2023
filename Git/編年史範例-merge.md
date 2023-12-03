## git merge (p.137)
合併分支 (須先切回主線)
```
$ git checkout master     # 若當前分支不在主線, 須先切回主線
$ git merge cat           # 合併分支
$ git merge cat --no-ff   # 若不複雜(快轉模式),硬是留下支線圖(有必要嗎?)
```
