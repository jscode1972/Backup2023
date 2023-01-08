/******************************************************************************************
* 說明: 建構模型 (model)
* 官網: https://mongoosejs.com/docs/
*      https://mongoosejs.com/docs/4.x/docs/guide.html
* 日期: 2019/03/14 過一個年都忘了, 複習 & 重整程式
------------------------------------------------------------------------------------------
// 有以下幾種型別: Array / Boolean / Buffer / Date / Mixed (A generic / flexible data type)
//               Number / ObjectId / String
******************************************************************************************/   
let mongoose = require('mongoose');
let emailSchema = new mongoose.Schema({
    pid : {
        type: String,
        required: true,
        unique: true 
    },
    name: String,
    email: String
});
// for instance method (需創建物件 new)
emailSchema.methods.findSimilarTypes1 = function(cb) {
    console.log(this);
};
// for class method (不需創建物件)
emailSchema.statics.findSimilarTypes2 = function(cb) {
    console.log(this);
};
module.exports = mongoose.model('Email', emailSchema);
