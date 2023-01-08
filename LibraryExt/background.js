// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

'use strict';

// 大鬍子影片 11.4 時間 8:36 此範例OK, 
// 情境一  popup => background 
// 發送方  BrowserAction 按下就會觸發 (無 popup.html) 
// 監聽方  background 需建立監聽器
chrome.browserAction.onClicked.addListener(browserClicked);       
function browserClicked(tab) {
   // 這裡會顯示在背景模式
   console.log('browserAction 被按下了, 在 [ ' + tab.title + ' ]');  
   // 情境二   background => content (進階使用)   
   let msg = { txt : "background 轉送訊息" };
   chrome.tabs.sendMessage(tab.id, msg); 
   // 監聽方  content 需建立監聽器 
   // 參見 content.js
};

// 1.2 當popup.html 按下觸發

// 2.2 發送訊息至 content_scripts


// 只要開啟分頁就有
// alert('background.js is loaded.');


/********************************
chrome.runtime.onInstalled.addListener(function() {
   chrome.storage.sync.set({color: '#3aa757'}, function() {
     //console.log("The color is green.");
   });

   chrome.contextMenus.create({ type : 'normal', title : 'Menu A', id : 'a' });
   chrome.contextMenus.create({ type : 'radio',  title : 'Menu B', id : 'b' });
   chrome.contextMenus.create({ type : 'radio',  title : 'Menu C', id : 'c' });
}); 
**********************************/
