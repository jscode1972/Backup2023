## git log
檢視紀錄
```
$ git log                                            # 內容太多,分頁檢視
$ git log --oneline --graph                          #
$ git log --graph --pretty=format:"%h <%an> %ar %s"  # 格式化
$ git log --oneline --author="Ben"                   # 找作者
$ git log --oneline --author="Ben\|Eddie"            # A或B, 分隔字元 \|
$ git log --oneline --grep="關鍵字"                   # 搜尋哪些 commit 說明有提到"關鍵字"
$ git log -S "關鍵字"                                 # 搜尋哪些 commit 檔案內容有提到"關鍵字"
$ git log --oneline --since="4pm" --until="6pm"      # 搜尋時間 
$ git log --oneline --after="2023-11"                # 搜尋日期 (可搭配時間,注意條件排序先後)
```

(範例一) git log --oneline --graph 
```
$ git log --oneline --graph

* f468e87 (HEAD -> master) 直接add+commit
* d6fb09e 空的2
* 9c7e047 空的
* b03b86e 1912-創建中國民國
* 1348b95 創建編年史
```

(範例二)
```
$ git log --graph --pretty=format:"%h <%an> %ar %s"

* f468e87 <BenHuang> 9 分鐘前 直接add+commit
* d6fb09e <BenHuang> 14 分鐘前 空的2
* 9c7e047 <BenHuang> 14 分鐘前 空的
* b03b86e <BenHuang> 2 小時前 1912-創建中國民國
* 1348b95 <BenHuang> 2 小時前 創建編年史
```
