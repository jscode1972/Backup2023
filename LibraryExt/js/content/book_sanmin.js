// *********************************************************************
// 博客來書店  http://www.books.com.tw/products/0010610362?loc=P_asb_002
// 書名:  $(".ProdName.purple16").text() // 電腦類可用
// 作者:  $("ul.ProdInfo").has("span#ISBN13").children(":nth-child(4)").children("a").text()
// 出版:  $("ul.ProdInfo").has("span#ISBN13").children(":nth-child(3)").children("a").text()
// 價格:  $("ul.PriceContent li:nth-child(2) span:nth-child(2)").text()
// ISBN: $("#ISBN13").parent().text().replace("ISBN13：","")
// *********************************************************************
// 命名規則: $Xyz(元素), oXyz(object), xXyz(文字), pXyz(parameter?)
// *********************************************************************

function SanminProducts() {
	// 若沒跑進來, 可能是 content_scripts
    oBook = {}; // 弄成廣域變數
    var $title = $(".ProdName.purple16"); // 電腦類可用
    var $btn = $("<a><span class='BtnPage'><button>收藏建檔</button></span></a>"); 
	//
	oBook.pathname = location.pathname;
    oBook.title = $($title).text();
	oBook.ISBN  = $("li.ga:contains('ISBN13：')").text().replace("ISBN13：","");
    oBook.author = $("ul.ProdInfo").has("span#ISBN13").children(":nth-child(4)").children("a").text();
    oBook.publisher = $("ul.ProdInfo").has("span#ISBN13").children(":nth-child(3)").children("a").text();
    oBook.from  = '三民';
    oBook.price = $("ul.PriceContent li:nth-child(2) span:nth-child(2)").text();
    //
    $($btn).attr({ pathname:  location.pathname,
                   title:     oBook.title,
    	           ISBN:      oBook.ISBN,
                   author:    oBook.author,
				   publisher: oBook.publisher,
                   price:     oBook.price }).click(SanminPost); 
    $($title).append($btn); 
    console.log('三民-done');
}

function SanminPost() {
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