## Codepen 測試專案 
* 專案路徑 https://codepen.io/jscode1972/pen/gOrrEeZ?editors=1010
* 確定可正常寫入 firestore (2020/08/18)

### HTML 區塊
```
<!-- The core Firebase JS SDK is always required and must be listed first -->
<script src="https://www.gstatic.com/firebasejs/7.18.0/firebase-app.js"></script>

<!-- TODO: Add SDKs for Firebase products that you want to use
     https://firebase.google.com/docs/web/setup#available-libraries -->
<script src="https://www.gstatic.com/firebasejs/7.18.0/firebase-auth.js"></script>
<script src="https://www.gstatic.com/firebasejs/7.18.0/firebase-firestore.js"></script>

<button onclick="storedata()">按鈕</button>
```
### JavaScript 區塊
```
// Your web app's Firebase configuration
var firebaseConfig = {
    apiKey: "AIzaSyAQXlw9rs_0cjsdzUCN1ZnwSWDBYhKfUC0",
    authDomain: "test-20200816.firebaseapp.com",
    databaseURL: "https://test-20200816.firebaseio.com",
    projectId: "test-20200816",
    storageBucket: "test-20200816.appspot.com",
    messagingSenderId: "92488528406",
    appId: "1:92488528406:web:42e640f51ed17be91b3b37"
};
// Initialize Firebase
firebase.initializeApp(firebaseConfig);
// 
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
