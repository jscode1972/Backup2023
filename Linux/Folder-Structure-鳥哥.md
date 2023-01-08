#### Linux 目錄結構, 參考 =>[鳥哥 第五章](http://linux.vbird.org/linux_basic/0210filepermission.php)
* bin 常用指令 (放置的是在單人維護模式下還能夠被操作的指令)
* boot 開機/核心 (開機會使用到的檔案)
* dev 設備 (任何裝置與周邊設備都是以檔案的型態存在於這個目錄當中的)
* etc 設定擋 (系統主要的設定檔)
  * opt 必要 (第三方協力軟體設定檔 )
  * x11 ( X Window )
  * sgml 
  * xml
* lib 開機會用到的函示庫 ( 以及 /bin & /sbin 會用到)
* media 媒體 (可移除的裝置, 軟碟、光碟、DVD)
* mnt 掛載某些額外的裝置
* opt 第三方協力軟體放置的目錄
* run
* sbin 放在/sbin底下的為開機過程中所需要的，裡面包括了開機、修復、還原系統所需要的指令
* src 網路服務啟動之後，這些服務所需要取用的資料目錄
* tmp 正在執行的程序暫時放置檔案的地方,開機會刪除
* usr => 第二層 FHS 設定
  * unix
  * software
  * resource
* var => 第二層 FHS 設定, 為放置變動性的資料，

### ★ 比較
|   | 可分享 (shared) | 不可分享 |
|:------------------|:-----------------:|----------------:|
|不變 (static)       |/usr/ 軟體源碼      |/etc  設定        |
||/opt/ 第三方                           |/boot/ 開機/核心   |
|可變 (variable)     |/var/mail/ 郵件     |/var/run/        |
||/var/spool/ 新開                       |/var/lock/       |
