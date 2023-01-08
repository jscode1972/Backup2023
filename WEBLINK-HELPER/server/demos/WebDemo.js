/******************************************************************************************
* 說明: 操作模型練習 (manipulate) 可正式啟用
* 官網: https://mongoosejs.com/docs/
*      https://mongoosejs.com/docs/4.x/docs/guide.html
* 日期: 2019/03/14 過一個年都忘了, 複習 & 重整程式
******************************************************************************************/   
var Database = require('../models/database'); // 只要入口宣告即可
var WebLinkModel = require('../models/WebLinkModel');
// 設定日期
let now = (new Date()).toLocaleString();

/************************************************************
// 1. class method
// WebLinkModel.classMethod({ link : 'classMethod'});
// 2. instance method
// var web = new WebLinkModel({ url : 'http://google.com' });
// web.instanceMethod({ link : 'instanceMethod'});

//************************************************************/
// 法一 存檔示範(自訂方法) 
x = { url: 'https://abc.com', title: 'ABC', tags: ['新聞3','科技2'], time: now };
WebLinkModel.CustSave(x);
return;

//************************************************************/
// 法二 存檔示範(官方API), => 部分取代 (不會異動沒有指定的部分) 
var query1 = { url: 'https://udn.com' }; 
var update1 = { $set: { url: 'https://udn.com', title: '中國時報', tags: ['新聞','科技', '購物', '政治'], time: now } };
var opts1 = { upsert: true }; // 找不到就新增
WebLinkModel.findOneAndUpdate(query1, update1, opts1, function(err, res) { // res 代表 resolves 
  if (err) console.log(err); // 假如失敗
  if (res) console.log(res); // 假如成功
}); 
return;

//************************************************************/
// 法三 存檔示範(官方API), => 全部取代 (沒有指定都會不見)
var query2 = { url: 'https://www.tnml.tn.edu.tw/' }; 
var update2 = { $set: { url: query2.url, title: '台南總圖', keywords: '圖書館,藏書', time: now  } };
WebLinkModel.findOneAndReplace(query2, update2, function(err, res) { // res 代表 resolves 
  if (err) console.log(err); // 假如失敗
  if (res) console.log(res); // 假如成功
});
//************************************************************/