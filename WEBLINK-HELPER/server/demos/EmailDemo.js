/******************************************************************************************
* 說明: 操作模型練習 (manipulate) 可正式啟用
* 官網: https://mongoosejs.com/docs/
*      https://mongoosejs.com/docs/4.x/docs/guide.html
* 日期: 2019/03/14 過一個年都忘了, 複習 & 重整程式
******************************************************************************************/   
var Database = require('../models/database'); // 只要入口宣告即可
let EmailModel = require('../models/EmailModel');
var email = new EmailModel({ pid: 'xxx', name: 'aaa', email:'ha@abc' });
console.log('object ==========================================');
//email.findSimilarTypes1('aa');
console.log('class ==========================================');
EmailModel.findSimilarTypes2('cc');
/*
let msg = new EmailModel({
  pid: 'D1203333',
  name: '頻果電腦',
  email: 'xxx@tsmc.com'
});
// https://medium.freecodecamp.org/introduction-to-mongoose-for-mongodb-d2a7aa593c57
// 這種寫法會重複寫入
msg.save()
   .then(doc => {
     console.log(doc)
   })
   .catch(err => {
     console.error(err)
   });
*/