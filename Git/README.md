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
- 語法
  - 平行 jobs
  - stages
  - whoami, hostname, uname
  - needs
  - before
  - only / except
  - tags

[和艦長一起 30 天玩轉 GitLab]:https://gitlab-book.tw/
