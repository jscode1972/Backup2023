/******************************************************************************************
* 說明: 將 mongoose.Schema 打包成模型類別 (純測試)
* 官網: https://mongoosejs.com/docs/
*      https://mongoosejs.com/docs/4.x/docs/guide.html
* 範例: 1.上半部弄成 module.exports  2.下半部引用呼叫
******************************************************************************************/
var db = require('../models/database'); // 只要入口宣告即可
// schema + class => model 
var mongoose = require('mongoose');  
const userMeta = { emplid: { type : String, required: true }, 
                   name: String, 
                   email: String };
const userSchema = new mongoose.Schema(userMeta);
class UserClass {
    doSave() { this.save().then(this.doSucc).catch(this.doFail); } // promise call
    doSucc(result) { console.log(result); } // promise
    doFail(error) { console.log(error); }   // promise
}
userSchema.loadClass(UserClass);
const UserModel = mongoose.model('User', userSchema);  // User => collection name users
// module.exports = UserModel; // 若是自成一個單元, 可匯出

// 應用 法一 (透過 model 更新)
var query = { emplid: "0002" };
var update = { $set: { emplid: '0002', name: '愛因斯坦', email: 'einstein@usa.com' } };
var opts = { upsert: true }; // 此參數很重要,若沒有資料則新增
UserModel.findOneAndUpdate(query, update, opts, function(err, res) {
  if (err) { console.log(err); }; // promise
  if (res) { console.log(res); }; // promise
});

// 應用 法二 (個體單純寫入.沒有更新機制)
/***************************** 
var user = new UserModel({emplid: '0002', name: 'XiJingPing' } );
user.email = 'xi@gmail.com';
user.doSave();
***************************** */
