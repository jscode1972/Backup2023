## gcloud 指令
* [gcloud CLI總覽 (官網文件)](https://cloud.google.com/sdk/gcloud/)
* [init](#init) 初始化 SDK    [參考](https://cloud.google.com/sdk/docs/initializing)
* [auth](#auth) 授權 SDK 工具 [參考](https://cloud.google.com/sdk/docs/authorizing)
* [config](#config)
* [info](#info)

--------------------------------
### init
初始化 SDK
```
$ gcloud init                  # 初始化
$ gcloud init --console-only   # 若在遠端,請下這一行就不會出現瀏覽器
```

### auth
授權 SDK (需要啟用服務帳戶)
```
$ gcloud auth login                       # 切換登入帳號, Authorize with a user account without setting up a configuration.
$ gcloud auth login --no-launch-browser   
$ gcloud auth activate-service-account [ACCOUNT] --key-file=[KEY_FILE]
$ gcloud auth list                       # 查看當前已授權帳戶 (ACTIVE) 
$ gcloud auth revoke [ACCOUNT]           # 收回授權
```

### config
```
$ gcloud config list
$ gcloud config list project              # 查看當前專案 (包含已刪除的)
$ gcloud config set project gcp-20200913  # 更換專案指令
$ gcloud config set account [ACCOUNT]     # 切換帳號 [ACCOUNT] is the full e-mail address of the account.
```

### info
```
$ gcloud info                            # Finding your credential files
```
