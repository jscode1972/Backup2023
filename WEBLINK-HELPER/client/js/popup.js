// 準備 popup 畫面
function showLink() {
    var query = { active: true, currentWindow: true };
    function callback(tabs) {
        var currTab = tabs[0]; // there will be only one in this array
        //console.log(currTab); // also has properties like currentTab.id
        $("#title").val(currTab.title);
        $("#urlFull").text(currTab.url);
        var matches = currTab.url.match(/:\/\/(.[^\/]+)(.*)/);
        $("#urlHost").text(matches[1]);
        // 查詢資料庫
        download();
    }
    // 讀取目前頁籤資料 (網址,標題,主機)
    chrome.tabs.query(query, callback);
}
// 準備標籤選項
function showTags() {
    // 主題 checkbox
    $.each(allTags.topic, function (key, data) {
        var row = $('<div class="row">').addClass("alert-success");
        checkInRow(row, key, data);
        $("#content").append(row);
    });
    // attribute checkbox
    var row = $('<div class="row">').addClass("alert-primary"); 
    checkInRow(row, '特性', allTags.attr);
    $("#content").append(row);
    // priority radio
    var row = $('<div class="row">').addClass("alert-danger"); 
    radioInRow(row, '優先', allTags.prio, 'prio');
    $("#content").append(row);
}
// 
function checkInRow(row, key, data){
    // head
    var head = $('<div class="col-sm-2">').text('['+key+']');
    row.append(head);
    // content
    var arr = [];
    data.forEach(function(tag) {
        var chk = $('<input type="checkbox">').val(tag);
        var span = $('<span>').text(tag);
        var div = $('<div class="col-sm-2">');
        div.append(chk, '&nbsp;', span);
        arr.push(div);
    });
    row.append(arr);
}
// 
function radioInRow(row, key, data, grp){
    // head
    var head = $('<div class="col-sm-2">').text('['+key+']');
    row.append(head);
    // content
    var arr = [];
    data.forEach(function(tag) {
        var rdo = $('<input type="radio">').val(tag).attr('name',grp);
        var span = $('<span>').text(tag);
        var div = $('<div class="col-sm-2">');
        div.append(rdo, '&nbsp;', span);
        arr.push(div);
    });
    row.append(arr);
}
// chrome extension 不允許 html 直接綁定 onclick, 故需指定 listener
document.addEventListener('DOMContentLoaded', function() {
    var btn = document.getElementById('btn');
    // onClick's logic below:
    btn.addEventListener('click', function() {
        upload();
    });
});
// 下載
function download() {
    getTag($("#urlFull").text()).then(function(data){
        $("#title").val(data.title);
        data.tags.forEach(function(x){
            $('input:checkbox[value="'+ x +'"]').prop('checked',true);
            $('input:radio[value="'+ x +'"]').prop('checked',true);
        });
        $("#keywords").val(data.keywords);
    }).catch(function(err){
        //
    });
}
// 上傳
function upload() {
    var arr = [];
    $('input:checked').each(function() { // 不限 checkbox 含 radio
        // 需判斷是否選取 
        arr.push($(this).val());
    });
    arr.forEach(function(a){
        var row = $('<div class="row">').text(a);
        $("#footer").append(row);
    });
    //
    var tagData = { url: $("#urlFull").text(), 
                    title: $("#title").val(), 
                    tags: arr,
                    keywords: $("#keywords").val()
                  };
    setTag(tagData).then(function(data) {
        alert('ok');
    }).catch(function(err) {
        console.log(err);
    });
};

// vscoe 快速指令 indent/..
// https://code.visualstudio.com/shortcuts/keyboard-shortcuts-macos.pdf
showLink();
showTags();

//window.open('http://google.com');
