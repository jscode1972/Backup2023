/******************************************************************************************
* 打包: 將 mongoose 打包成單一類別, 放到主程式直接引用即可
* 官網: https://mongoosejs.com/docs/
*      https://mongoosejs.com/docs/4.x/docs/guide.html
* 術語: Collections => 表格 => tables
*      Documents   => 紀錄 => records or rows
*      Fields      => 欄位 => columns
*      Schema      => 定義 => table definition, document data structure (or shape of the document) 
*      Models(*)   => higher-order constructors that take a schema and create an instance of 
*                     a document equivalent to records in a relational database.
******************************************************************************************/    
var mongoose = require('mongoose');
var config = require('./config');   // 密碼設定檔
// 帶入密碼檔參數
const account  = config.account;    // REPLACE WITH YOUR DB ACCOUNT
const password = config.password;   // REPLACE WITH YOUR DB PASSWORD
const server   = config.server;     // REPLACE WITH YOUR DB SERVER
const database = config.database;   // REPLACE WITH YOUR DB NAME
// 負責初始連線物件
class Database {
  // 建構式
  constructor() {   
    this._connect();  
  }
  // 初始連線
  _connect(){
    var url = `mongodb+srv://${account}:${password}@${server}/${database}`;
    var opts = { useNewUrlParser: true };
    mongoose.connect(url, opts);
    var conn = mongoose.connection;
    conn.on('error', console.error.bind(console, 'connection error:'));
    conn.once('open', function() { console.log('open'); });
  }
}
// 輸出, 只會執行一次 (只有一個實例)
module.exports = new Database();

/****************************************** 
  _connect2() { // 此法亦可
     mongoose.connect(`mongodb+srv://${account}:${password}@${server}/${database}`,
        {useNewUrlParser: true})
       .then(() => {
         console.log('Database connection successful');
       })
       .catch(err => {
         console.error('Database connection error');
       });
  }  
  *************************************/