## git reset
拆掉最後兩次 commit
```
$ git reset HEAD^^       # 回復前兩次 commit
$ git reset HEAD~2       # 回復前兩次 commit (等效)
$ git reset master^^     # 回復前兩次 commit (等效)
$ git reset 3a010f0^^    # 回復前兩次 commit (等效) 3a010f0 當前 commit   (相對)
$ git reset 21b9746^     # 回復前兩次 commit (等效) 21b9746 為前一次commit (相對)
$ git reset b3bdb0a      # 回復前兩次 commit (等效) b3bdb0a 為前二次commit (直接指定)
```
