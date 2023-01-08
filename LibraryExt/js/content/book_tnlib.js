// *********************************************************************
// 台南圖書館  https://www.tnml.tn.edu.tw/ 
// 我的書房 > 借閱紀錄 > 已借書   
//   http://163.26.71.106/webpac/shelf_borrow.cfm
//   借書欄頭 *[@id="fun_top"]/div
//   借書列表 *[@id="list"]
//   功能列表 *[@id="list"]/div[1]/           => 先刪掉                       
//   第一本書 *[@id="list"]/div[2]/  (1/20筆) => div[1] (考慮改用 list_box?)
//   第二本書 *[@id="list"]/div[3]/  (2/20筆) => div[2]
//                        /listbox/title 
//                        /listbox/content
// *********************************************************************
// 命名規則: $Xyz(元素), oXyz(object), xXyz(文字), pXyz(parameter?)
// *********************************************************************

function TainanShelfBorrow() {
    //$("body").append("<div id='dialog' style='display:none;''>dialog</div>");
    // binding button
    var btn = $("<a class='btn_small'>收藏建檔</a>"); // 沒問題
    $(btn).click(TainanBatch); 
    $("#fun_top > div").append(btn);             // 沒問題
    // 第一個 div 先刪掉, 讓 child 正常流水號
    $("#list").children("div:first").remove();
    // 每本書都加上再借按鈕
    $('#list').children().each(function(index) { // <a class='btn_small' id='button'>續借</a>
        /********************************/
        var href = $("div:first > ul li:nth-child(3) a", $(this)).attr("href"); // content.cfm?mid=666057&m=ss
        var $content = $("div.product_list_content", $(this));
        var barcode = myReplace($("div:first p:nth-child(1)", $($content)));
        var mid = href.replace('content.cfm?mid=','').match(/\d+/i);
        var star = $("<a class='btn_small' id='button'>好</a>");
        var must = $("<a class='btn_small' id='button'>借</a>");
        var read = $("<a class='btn_small' id='button'>讀</a>");
        $(star).attr('mid',mid).attr('barcode',barcode).css("background-color","#86af49").click({flag:'star'},TainanMark);
        $(must).attr('mid',mid).attr('barcode',barcode).css("background-color","#86af49").click({flag:'must'},TainanMark);
        $(read).attr('mid',mid).attr('barcode',barcode).css("background-color","#86af49").click({flag:'read'},TainanMark);
        $("div:first > ul li:nth-child(3)", $(this)).append(star);   
        $("div:first > ul li:nth-child(3)", $(this)).append(must);  
        //$("div:first > ul li:nth-child(3)", $(this)).append(read); // 太長塞不下 
        /********************************
        ////*[@id="list"]/div[1]/div[2]/div[2]/div[1]/div[1]/p[1]
        // $("#list .list_box:first .product_list_content .info:first p:first") ok
        // $("#list .list_box:first .product_list_content p.info_long:first") ok
        // $("#list .list_box:first p.info_long:first")*/
    });
}

function TainanBatch() {
    var $list = []; // 多筆批次
    //$('#list .product_list_content').each(function(index) {
    $('#list').children().each(function(index) {
        var $book = {};
        var $title = $("div ul li:nth-child(3) a", $(this));
        var $content = $("div.product_list_content", $(this));
        // 條碼號：31204004845922, 借出日：2018/03/25, 索書號：412.51 4447 2011, 到期日：
        $book["title"]    = $($title).text(); // $("div ul li:nth-child(3) a", $(this)).text();
        // title, console.log(title);
        /*if ($book["title"].indexOf('/') > -1) {
        	$book["title"] = $book["title"].substring(0, $book["title"].indexOf('/'));
        }*/
        $book["title"]    = trimTill($book["title"], '/');
        // content (標題要前後對應大小寫)
        $book["mid"]      = String($($title).attr("href").match(/\d+/));
        $book["barcode"]  = myReplace($("div:first p:nth-child(1)", $($content))); 
        $book["lend"]     = myReplace($("div:first p:nth-child(2)", $($content))); 
        $book["callno"]   = myReplace($("div:nth-child(2) p:nth-child(1)", $($content))).match(/\S+\s\S+/); 
        $book["due"]      = myReplace($("div:nth-child(2) p:nth-child(2)", $($content))); 
        $book["location"] = myReplace($("div:nth-child(2) p:nth-child(5)", $($content)));  
        // console.log( index + ": " + JSON.stringify( $book ) );
        $list.push($book);
    });
    //  alert($(".product_list_content", $(this)).innerText);
    TainanPost($list);
}

// 目前無用
function TainanMark(par) {
    // 不論比數, 共用 API, 所以用 list
    var $list = [];  // 型為多筆, 實為單筆
    var $book = {};
    $book["mid"]  = $(this).attr('mid');
    $book["barcode"]  = $(this).attr('barcode');
    $book[par.data.flag] = "v"; // flag => star, must
    $list.push($book);
    //
    TainanPost($list);
}

function TainanPost($list) {
    var $par = { svc : 'book',
                 action : 'TainanUpdate', 
                 list : JSON.stringify($list)
               };
    $.post($ggapi, $par, function(text, status) { 
        var json = JSON.parse(text);
        if (json["status"] == "ok") {
            console.log(json);
            alert('ok');
        } else {
            alert("fail");  
        }
    });
}


function KKDialog(msg) {
  $('#dialog').html(msg);
  $('#dialog').dialog({
      autoOpen: true,
      show: "blind",
      hide: "explode",
      modal: true,
      open: function(event, ui) {
          setTimeout(function(){
              $('#dialog').dialog('close');                
          }, 3000);
      }
  });
}


