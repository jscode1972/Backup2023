// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

function myAlert() {
	//alert('x3');
	//chrome.tabs.create({url: chrome.extension.getURL('background.html')});

    // 下例不能用, 改 query
    // chrome.tabs.getCurrent(gotTab);
    let params = { active : true, currentWindow : true};
    chrome.tabs.query(params, gotTabs);
    //
    function gotTabs(tabs) {
      let msg = { txt : "hello from popup.js" };
      chrome.tabs.sendMessage(tabs[0].id, msg); 
      
    }
    // manifest.json 可覆蓋原本 
    // "chrome_url_overrides" : {
    // "newtab": "background.html"
    //  }
}

function YoutubePost() {
    var oLink = { href : $(this).attr('href'), 
                  title : $(this).attr('title'),
                  tags : ""
                };
    var oPar = { svc : "book",  
                action : "YoutubeUpdate", 
                link : JSON.stringify(oLink)
              };
    $.post($ggapi, oPar, function(text, status) {
        var json = JSON.parse(text);
        if (json["status"] == "ok") {
            console.log(json); 
            //alert("ok"); 
        } else {
            alert("fail");  
        }
    });
}

$("#test").click(myAlert);

// 改標題, Badge
chrome.browserAction.setTitle({ title : '小幫手' }); 
chrome.browserAction.setBadgeBackgroundColor({ color : '#ff0000' }); 
//chrome.browserAction.setBadgeText({ text : '999+' }); 

// 安裝在 onInstalled
//chrome.contextMenus.create({ type : 'normal', title : 'Menu A', id : 'a' });
//chrome.contextMenus.create({ type : 'radio',  title : 'Menu B', id : 'b' });
//chrome.contextMenus.create({ type : 'radio',  title : 'Menu C', id : 'c' });

$(document).ready(function() {
	chrome.tabs.getSelected(null, function(tab) {
        if (tab.url.match(/youtube.com/i) == 'youtube.com') {
            $btn = $("<button type='button'>加入Youtube</button>")
              .attr("title", tab.title)
              .attr("href", tab.url)
              .click(YoutubePost);
            $("#main").append($btn);
        } else {
		        var link = document.createElement('a');
		        link.href = tab.url;
		        link.innerText = 'xxx';
		        $("#host").html('html: ' + link.hostname + ' <br>' + location.href + '<br>' + link.href);
        }
	});
});