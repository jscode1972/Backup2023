## 平台測試SDK @CodePen

#### CodePen 免費線上平台  
* codepen https://codepen.io/
* 使用 github 登入, 帳號 jscode1972@
* 創建 new pen  https://codepen.io/jscode1972/pen/gOrrEeZ?editors=1010
* HTML 段落
  * 參考 SDK 語法範例
  * `<script src="https://www.gstatic.com/firebasejs/7.18.0/firebase-app.js"></script>`
  * `<script src="https://www.gstatic.com/firebasejs/7.18.0/firebase-auth.js"></script>` (SDK 語法範例有參考網址)
  * `<script src="https://www.gstatic.com/firebasejs/7.18.0/firebase-firestore.js"></script>` (SDK 語法範例有參考網址)
  * `<button onclick="storedata()">按鈕</button>` (自己補充)
* 引用宣告
  ```
   var firebaseConfig = {
     apiKey: "AIzaSyAQXlw9rs_0cjsdzUCN1ZnwSWDBYhKfUC0",     // 調用某些不需要訪問私有用戶數據的API時使用的簡單加密字符串
     authDomain: "test-20200816.firebaseapp.com",
     databaseURL: "https://test-20200816.firebaseio.com",   // 實時數據庫URL  database
     projectId: "test-20200816",                            // 整個Firebase和GCP中項目的用戶定義的唯一標識符。
     storageBucket: "test-20200816.appspot.com",            // 雲存儲存儲桶   storage
     messagingSenderId: "92488528406",
     appId: "1:92488528406:web:42e640f51ed17be91b3b37"      // Firebase應用程序在所有Firebase中的唯一標識符，具有特定於平台的格式：
   };
   var firebase = window.firebase;
  ```
* 參數說明
  * Firebase iOS應用程序： GOOGLE_APP_ID （示例值： 1:1234567890:ios:321abc456def7890 ）這不是 Apple捆綁包ID。
  * Firebase Android應用程序： mobilesdk_app_id （示例值： 1:1234567890:android:321abc456def7890 ）這不是 Android包名稱或Android應用程序ID。
  * Firebase Web應用程序： appId （示例值： 1:65211879909:web:3ae38ef1cdcb2e01fe5f0c ）
* JavaScript 段落
  * 參考 SDK 語法範例
  * 補充讀寫 db 語法
    ```javascript
    var db = firebase.firestore();
    function storedata() {
      db.collection("movies").doc("新世紀福爾摩斯").set({
        name: "新世紀福爾摩斯",
        date: "201022",
        desctiption: "本劇改編自阿瑟·柯南·道爾爵士家喻戶曉的推..。",
        actors: ["班尼迪克·康柏拜區", "馬丁·費曼"]
      });
    }
    ```
* 寫入成功
 * 主控台/Database/多一個 movie collection
