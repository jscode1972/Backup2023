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

function UnipProducts() {
    // 若沒跑進來, 可能是 content_scripts
    oBook = {}; // 弄成廣域變數
    var $title = $("div.type02_p002 h1");
    var $btn = $("<a class='type02_btn05'><span>收藏建檔</span></a>"); 
	  //
    oBook.pathname = location.pathname;
    oBook.title = $title.text();
    oBook.ISBN  = $("li:contains('ISBN')").text().replace('ISBN：','');
    oBook.author = $("#trace_box").next().text();
    oBook.publisher = $(".type02_p003 ul li:nth-child(3) a:first span").text();
    //oBook.author = $("li:contains('作者：')").text().replace('作者：','');        // 需再過濾
    //oBook.publisher = $("li:contains('出版社：')").text().replace('出版社：','');  // 需再過濾
    oBook.from  = '博客來';
    oBook.price = $("strong.price01 b").text();
    //
    $($btn).attr({ pathname:  location.pathname,
                   title:     oBook.title,
                   ISBN:      oBook.ISBN,
                   author:    oBook.author,
                   publisher: oBook.publisher,
                   price:     oBook.price }).click(UnipPost); 
    $($title).append($btn); 
}

function UnipPost() {
	//alert(JSON.stringify(oBook));
	var $par = { svc : 'book',
               action : 'MustUpdate', 
               book : JSON.stringify(oBook)
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