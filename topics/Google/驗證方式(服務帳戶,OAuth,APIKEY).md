## 驗證方式有點搞不懂
* [以服務帳戶進行驗證](https://cloud.google.com/docs/authentication/production?hl=zh-tw)
  * 如果在 google 環境外面, 需創建金鑰並下載引用
* [以使用者身份進行驗證](https://cloud.google.com/docs/authentication/end-user?hl=zh-tw)
  * OAuth 2.0.
* [使用 API 金鑰](https://cloud.google.com/docs/authentication/api-keys?hl=zh-tw)
  * 一組簡單編碼的字串, 不要放在程式碼裡, 需放到設定檔
  * 若需傳送以 key=API_KEY 格式傳送
