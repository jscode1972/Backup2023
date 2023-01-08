// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// 此範例 OK, 但是要開啟正常網頁, 設監聽器接收 message
// 1. 當BrowserAction按下就會觸發
// 2.1 background 接收來自 browserAction 按鈕訊息
// 2.2 background 發送訊息至 content_scripts
// 3. 接收來自 background 訊息
chrome.runtime.onMessage.addListener(gotMessage);
function gotMessage(msg, sender, sendResponse) {
	// 這裡的 console 確實是網頁的, 不是後端
	console.log(sender);  // 確定是 extension 軟件的 id
	console.log(msg.txt); // 顯示 OK
	// 再進階控制網頁變化 （博客來網頁) 
	$("#trace_btn1").remove();  // YES! 成功把追蹤作者刪除
}

// 真的只有載入網頁才會一併執行
//alert('content.js is loaded.');

/****** 判斷網頁路由 *********************************************/
var href = location.href;
//
if (href.match(/shelf_borrow.cfm/i) == 'shelf_borrow.cfm') {
   TainanShelfBorrow();
} else if (href.match(/products/i) == 'products') {
	UnipProducts();
} else if (href.match(/Product\/index/i) == 'Product/index') {
    SanminProducts();
} else if (href.match(/watch/i) == 'watch') {
    YoutubePages();
}
/**************************************************************/

//"scripts": ["background.js"],

/*****************************************************************************
// 增加button
$('#list').children().each(function(index) { // <a class='btn_small' id='button'>續借</a>
	var mark = $("<a class='btn_small' id='button'>必借</a>");
	$(mark).width('30px');
    $("div:first > div:first", $(this)).append(mark); 
});
****************************************************************************/

//$("a").click(myAlert); // 確定OK, 可用來測試上面程式有沒有問題
//$("div.button_login").click(myAlert); // 馬拉松
//$("a[title='館藏查詢']").click(myAlert); // 確定OK
//$("a").click(myAlert); // 確定OK