// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

$ggapi = 'https://script.google.com/macros/s/AKfycbxWCZt2k7uhyceZeCcfpJNF32SWmwu36J76Y6CYFMD_pBwDt8c/exec';

// 抓某個字元之前的資料
function trimTill(text, char) {
    var output = text;
    if (text.indexOf(char) > -1) {
       output = text.substring(0, text.indexOf(char));
    }
    return output;
}

function sleep(milliseconds) {
	var start = new Date().getTime();
	for (var i = 0; i < 1e7; i++) {
		if ((new Date().getTime() - start) > milliseconds){
			break;
		}
	}
}

// 取代台南圖書館的標籤文字
function myReplace(txt) {
    return $(txt).text()
                 .replace('條碼號：','')
                 .replace('借出日：','')
                 .replace('索書號：','')
                 .replace('到期日：','')
                 .replace('館藏地：','')
                 .replace('開架閱覽區','')
                 .replace('年度新書區','')
                 .replace('罕用書庫(需線上調閱)','')
                 .replace('ISBN：','');
}
