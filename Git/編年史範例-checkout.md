## git checkout (p.073)
切換分支 (HEAD 移動)
```
$ git checkout cat        # 切換分支
$ git checkout -b cat     # 切換分支 (強迫建立)
```

救回被刪掉的檔案 
- 也適用於修改檔案反悔,將之回覆
- 若已加入暫存區再度修改(此時1紅1綠),回覆的是暫存區資料
```
$ git checkout .                # 救回/回復所有檔案
$ git checkout 檔案              # 救回/回復指定檔案
```

救回最後兩次檔案 (回復前兩次檔案,範例等效)
```
$ git checkout HEAD^^ 檔案       # 回復前兩次檔案
$ git checkout HEAD~2 檔案       # 回復前兩次檔案 (等效)
$ git checkout master^^ 檔案     # 回復前兩次檔案 (等效)
$ git checkout 3a010f0^^ 檔案    # 回復前兩次檔案 (等效) 3a010f0 當前 commit   (相對)
$ git checkout 21b9746^ 檔案     # 回復前兩次檔案 (等效) 21b9746 為前一次commit (相對)
$ git checkout b3bdb0a 檔案      # 回復前兩次檔案 (等效) b3bdb0a 為前二次commit (直接指定)
```
