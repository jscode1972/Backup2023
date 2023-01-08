// 參考 alexa 網站分類
var allTags = {
  "topic": {   // 須嚴格定義主題是什麼
    "娛樂": ['閱讀','電影','影集','音樂'],
    "興趣": ['旅行','跑步','登山','健身'],
    "電腦": ['新知學習','開發工具','程式語言'],
    // 特別關注的重要主題
    "關注": ['機票','大洋路','好書'] 
  },
  "attr": ['工具', '教學', '文摘', '論壇', '社群'],
  "prio": ['置頂', '重要', '普通', '暫存']
};
// 
function getTag(tagUrl) {
  return new Promise(function(resolve, reject) {
    $.ajax({
      type :"GET",
      url: "http://localhost:8080/web/link",   //存取Json的網址    
      data : {  url : tagUrl },
      dataType: "json",
      success : function(data) {
        resolve(data);         
      },
      error: function (err) {
        reject(err);
      }
    });
  });
}
// 上傳資料
function setTag(tagData) {
  //console.log(tagData);
  // Return a new promise.
  return new Promise(function(resolve, reject) {
    $.ajax({
      type: "POST",
      url: "http://localhost:8080/web/link",   //存取Json的網址             
      data: tagData,
      cache: false,
      dataType: 'text', // 一定要正確, 不然就算 success, 都會跑錯 flow
      success: function (data) {
          resolve(data);
      },
      error: function (err) {
          reject(err);
      }
    }); 
  });
}