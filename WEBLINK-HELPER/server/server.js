/**********************************************************
方法	          說明
================  =========================================
res.download()	  提示您提供要下載的檔案。
res.end()	      結束回應程序。
res.json()	      傳送 JSON 回應。
res.jsonp()	      傳送 JSON 回應，並支援 JSONP。
res.redirect()	  將要求重新導向。
res.render()	  呈現視圖範本。
res.send()        傳送各種類型的回應。
res.sendFile      以八位元組串流形式傳送檔案。
res.sendStatus()  設定回應狀態碼，並以回應內文形式傳送其字串表示法。
***********************************************************/
var db = require('./models/database'); // 宣告即可
var express = require('express');
var app = express();   
var bodyParser = require('body-parser');
var web_route = require('./routes/web-route');
//
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
// 可載入靜態檔案
app.use(express.static('public'));
app.use('/web', web_route);
app.listen(8080, function () {
    console.log('Example app listening on port 8080!');
});
