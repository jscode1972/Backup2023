## gitlab 教學
- [和艦長一起 30 天玩轉 GitLab]


## .gitlab-ci.yml 
如前述會包含 “觸發條件”、”環境”、”步驟”
- 觸發條件: PR、PUSH、分支合併
- 環境: Node、Python、Java 等等
- 步驟: 常見會有三個步驟
  - Verify: 透過 Linter 或是測試確認程式碼品質
  - Package: Build Code
  - Release: APK 就會是包版、前後端就會是上版
- pipeline
  - https://gitlab.com/gitlab-org/gitlab/-/pipelines/166785558
- gitlab runner: 負責運行 pipeline 中 job
  - Shared Runner: 不同的專案可以共用
  - Group Runners: 同開發群組 (部門) 共用
  - Specific Runner: 指定給特定專案使用，小編在實務上也會透過 Specific Runner 來指定要發佈的環境

## 高見龍
- 01 介紹
  - job / stages / pipeline
  - 環境變數
  - 有條件執行
  - gitlab 架構
  - 部署
  - docker
- 04 建立 .gitlab-ci.yml
  - rsa login
  - script, before_script, after_script
- 05 工作階段以及相依性
  - 觀察 Web UI
  - 平行處理 (bad job)
  - stages
    -  (linter/testing/build/deploy) 前一階段沒過不跑
  - needs
    - 同一stage => bad job (相依job沒過不跑
- 06 執行外部檔案 (複雜指令/又臭又長/環境相關)
  - chmod 755 ./run.sh
  - ./run.sh
- 07 指定分支
  - only
  - except
- 08 環境變數 (回頭仔細看!!)
  - variables
    - local / golbal
    - pre-defined (https://docs.gitlab.com/ee/ci/variables/predefined_variables.html)
    - 專案變數 (不在本檔案定義)
  - workflow (比較難先跳過)
    - rules 精細控制
      - if
      - when
- 09 docker (回頭再看)
  - image (略..)
- 10 Runner 與 Executor
  - runnung with
  - cleanup...
  - 設定
    - 環境變數 
    - 執行器 runner
- 11 本機 runner
- 12 digital ocean
- 13 Git Runner
- 14 專案演練-1
- 15 專案演練-2
- 16 專案演練-3
- 17 專案演練-4
- jobs
  - whoami, hostname, uname
  - needs
  - before
  - only / except
  - tags

[和艦長一起 30 天玩轉 GitLab]:https://gitlab-book.tw/
