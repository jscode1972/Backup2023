// *********************************************************************
// 博客來書店  http://www.books.com.tw/products/0010610362?loc=P_asb_002
// 書名:  $("div.type02_p002 h1").text()
// 作者:  $("#trace_box").next().text()  或 $(".type02_p003 ul li:first > a:first").text()
// 出版:  $(".type02_p003 ul li:nth-child(3) a:first span").text()
// 價格:  $("strong.price01 b").text() 
// ISBN: $("li:contains('ISBN')").text().replace('ISBN：','')
// *********************************************************************
// 命名規則: $Xyz(元素), oXyz(object), xXyz(文字), pXyz(parameter?)
// *********************************************************************

function YoutubePages() {
    // 若沒跑進來, 可能是 content_scripts
    oLink = {}; // 弄成廣域變數
    var $title = $("div#container > h1");
    var $btn = $("<button>收藏建檔</button>"); 
	  //
    oLink.href = location.href;
    oLink.title = $title.text();
    oLink.tags = '';
    //
    $($btn).attr({ title:     oLink.title,
                   href:      oLink.href,   // 故意改成 pathname 不要跟原本 href 屬性衝突
                   tags:      oLink.tags }).click(YoutubePost); 
    $($title).append($btn); 
    console.log('youtube-done');
}

function YoutubePost() {
	//alert(JSON.stringify(oBook));
	var $par = { svc : 'book',  // 暫時棲身於此
               action : 'YoutubeUpdate', 
               link : JSON.stringify(oLink)
             };
    $.post($ggapi, $par, function(text, status) { 
    	var json = JSON.parse(text);
        if (json["status"] == "ok") {
            console.log(json); 
            //alert("ok"); 
        } else {
            alert("fail");  
        }
    });
}