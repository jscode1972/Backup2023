### Linux 常用指令
* 查看行程
  + ps aux | grep forever
* forever 
  + forever list
  + forever stopall
  + forever start server.js


### Windows 指令
* 比較 node_modules 新舊差異(新增)的部分,輸出為檔案 (方便複製到公司環境)
  + robocopy "c:\nodejs" "c:\nodejs_old" /e /l /ns /nc /ndl /njs /njh /fp /xx /log:trace.txt
    - /e 複製子目錄 (包含空目錄)
    - /l 只輸出清單 (不複製,不刪除,不加上戳記)
    - /ns 不記錄大小 (no size)
    - /nc 不記錄檔案類型 (no category)
    - /ndl 不記錄目錄清單 (比較 /nfl)
    - /njh 沒有工作表頭 (no job header)
    - /njs 沒有工作摘要 (no job summary)
    - /fp 輸出包含完整路徑 (full path)
    - /xx 排除其他檔案和目錄
    - /log:trace.txt
