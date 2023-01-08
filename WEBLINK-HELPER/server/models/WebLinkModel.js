/******************************************************************************************
* 說明: 建構模型 (model)
* 官網: https://mongoosejs.com/docs/
*      https://mongoosejs.com/docs/4.x/docs/guide.html
* 日期: 2019/03/14 過一個年都忘了, 複習 & 重整程式
------------------------------------------------------------------------------------------
// 有以下幾種型別: Array / Boolean / Buffer / Date / Mixed (A generic / flexible data type)
//               Number / ObjectId / String
******************************************************************************************/   
// 1.  Referencing Mongoose
let mongoose = require('mongoose');  // returns a "Singleton object"

// 2. Defining the Schema, (資料結構) 
const WebLinkMeta = {  
  url : { type: String, required: true, unique: true }, // 網址 (unique)
  title : { type: String, required: true },             // 網站名稱
  tags : [ { type: String } ],                          // 多重標籤 (全混在一起, 組織報表再另外設計物件)
  keywords : { type: String },                          // 關鍵字 'aaaa,bbb,ccc'
  time : { type: String }                               // 本地時間
};
let WebLinkSchema = new mongoose.Schema(WebLinkMeta);

// 3.  Customized methods (分成  1.類別靜態方法  2.實例物件方法)
// 3.1 類別靜態方法 class method (不需創建物件) 呼叫方式 => 
//     WebLinkModel.classMethod(參數);
WebLinkSchema.statics.classMethod = function(link) {
  //console.log(this);
};
// 3.2 實例物件方法 instance method (需創建物件 new) 呼叫方式 => 
//     var web = new WebLinkModel({ url : 'http://google.com' });
//     web.instanceMethod(參數);
WebLinkSchema.methods.instanceMethod = function(link) {
  //console.log(this);
};

// 客製化類別方法 Save
WebLinkSchema.statics.CustSave = function(link) {
  if (link.url) {
    var query = { url: link.url }; 
    var update = { $set: link };
    var opts = { upsert: true }; // 找不到就新增
    // findOneAndReplace 全部取代 (沒有指定都會不見)
    // findOneAndUpdate  部分取代 (不會異動沒有指定的部分) 
    this.findOneAndUpdate(query, update, opts, function(err, res) { // res 代表 resolves 
      if (err) console.log(err); // 假如失敗
      if (res) console.log(res); // 假如成功
    });
  }
};

// 4. Exporting a Model,
//   'WebLink' 勿亂改, 對應 collection "weblinks" (複數) 
// const WebLinkModel = mongoose.model('WebLink', WebLinkSchema); 
module.exports = mongoose.model('WebLink', WebLinkSchema);  
