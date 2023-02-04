索引表格 
=========================================
原檔存放 jscode1972/**Backup2023**
| 靠左------ | ------- 置中 ----- | ------- 置中 -----  | ---------------靠右 |
|:----------|:------------------:|:------------------:|-------------------:|            
| 區塊元素   | [引用文字](#引用文字) | [程式碼](#程式碼)    | 編修,2023/02/04     | 
| 列表元素   | [巢狀縮排](#巢狀縮排) | [diff](#diff)      | 編修,2023/02/04     | 
| 行內元素   | [字體變化](#字體變化) | [強迫跳行](#強迫跳行) | 編修,2023/02/04     | 
| 段落範例(*)| | | |
| 內部連結   | [頁內連結](#頁內連結) | [站內連結](#站內連結) | 編修,2023/02/04     | 
| 外部連結   | [外部連結](#外部連結) | [圖片連結](#圖片連結) | 編修,2023/02/04     |    
| 參考連結   | [官網][2]           | [Syntax][3]        | [Guide][4]         |

## 二級標題 (2個#,有底線)
#### 三級標題 (3~4個#)
###### 水平線 (5~6個#)  (連續3個 ***** 或  --- 或 underline)


## 區塊元素
#### 引用文字
    語法: 前面4空格, 但是必須在標題之後

#### 程式碼 
```javascript
```語言
var s = "高亮度程式碼";
alert(s);
```

[回到頂部(索引表格)](#索引表格)

## 列表元素
#### 巢狀縮排
列表可用符號: *+-123 
- 台北市 
- 台南市 
  - [x] 永康 `語法: - [x] xxxx`
  - [ ] 南區 
* 高雄市  
  1. 路竹 `語法: 1. 路竹`
  1. 茄萣

    內縮 => 上空一行 & 4 空格
    
        引用內縮 => 上空一行 + 8 空格
#### diff
綠色表示新增，紅色表示删除。
```diff
語法同程式碼, 標記 diff
+ 滷肉飯
- 魯肉飯
```

[回到頂部(索引表格)](#索引表格)

## 行內元素
#### 字體變化
斜體/粗體/行內反白
> *斜體 (1個星)*  
_斜體 (1底線)_  
**粗體 (2個星)**   
__粗體 (2底線)__  
Use the `行內反白` function.  
#### 強迫跳行 
兩個空格 ($$)
> 一個段落是由一個以上相連接的行句組成， $$  
而一個以上的空行則會切分出不同的段落  $$  
空行的定義是顯示上看起來像是空行，便會被視為空行。

[回到頂部(索引表格)](#索引表格)


## 段落範例
#### 區塊引言 (原始碼)
```
1.  區塊引言可以有階層（例如：引言內的引言），只要根據層數加上不同數量的>：
> This is the first level of quoting.
>
> > This is nested blockquote.
>
> Back to the first level.

2.  引言的區塊內也可以使用其他的Markdown語法，包括標題、清單、程式碼區塊等：
> ## This is a header.
> 
> 1.   This is the first list item.
> 2.   This is the second list item.
> 
> Here's some example code:
> 
>     return shell_exec("echo $input | $markdown_script");
```

#### 區塊引言 (效果)
a.  區塊引言可以有階層（例如：引言內的引言），只要根據層數加上不同數量的>：
> This is the first level of quoting.
>
> > This is nested blockquote.
>
> Back to the first level.

b.  引言的區塊內也可以使用其他的Markdown語法，包括標題、清單、程式碼區塊等：
> ## This is a header.
> 
> 1.   This is the first list item.
> 2.   This is the second list item.
> 
> Here's some example code:
> 
>     return shell_exec("echo $input | $markdown_script");


#### 清單包含多段落 (原始碼)
```
1.  清單項目可以包含多個段落，每個項目下的段落都必須縮排4個空白或是一個tab.

    Vestibulum enim wisi, viverra nec, fringilla in, laoreet
    vitae, risus. Donec sit amet nisl. Aliquam semper ipsum
    sit amet velit.

2.  如果你每行都有縮排，看起來會看好很多，當然，再次地，如果你很懶惰，Markdown也允許：
    
    This is the second paragraph in the list item. You're
only required to indent the first line. Lorem ipsum dolor
sit amet, consectetuer adipiscing elit.

3.  如果要在清單項目內放進引言，那>就需要縮排：

    > This is a blockquote
    > inside a list item.
4.  如果要放程式碼區塊的話，該區塊就需要縮排兩次，也就是8個空白或是兩個tab：

        <code goes here>
```

#### 清單包含多段落 (原始碼)
1.  清單項目可以包含多個段落，每個項目下的段落都必須縮排4個空白或是一個tab.

    Vestibulum enim wisi, viverra nec, fringilla in, laoreet
    vitae, risus. Donec sit amet nisl. Aliquam semper ipsum
    sit amet velit.

2.  如果你每行都有縮排，看起來會看好很多，當然，再次地，如果你很懶惰，Markdown也允許：
    
    This is the second paragraph in the list item. You're
only required to indent the first line. Lorem ipsum dolor
sit amet, consectetuer adipiscing elit.

3.  如果要在清單項目內放進引言，那>就需要縮排：

    > This is a blockquote
    > inside a list item.
4.  如果要放程式碼區塊的話，該區塊就需要縮排兩次，也就是8個空白或是兩個tab：

        <code goes here>
        

## 內部連結
#### 頁內連結
`[回到頂部](#文字元素)`
>[回到頂部](#文字元素)  
>[回到頂部](#a-bb-cc) 若有英文字, 空格用 -
#### 站內連結
`[連結名稱](/路徑/)`
>See my [About](/about/) page for details. 

[回到頂部(索引表格)](#索引表格)

## 外部連結
#### 外部連結
```
行內標記法: [Google](http://www.google.com/ "谷歌(hint)")
索引標記法: [谷歌][1]
// 文末索引
[google]: http://www.google.com 
[1]: http://www.google.com  "Google"
```
- [Google](http://www.google.com/ "谷歌")  `行內標記法`  
- [谷歌][1]       `索引標記法`  
#### 圖片連結
```
![Alt text][logo]
```
![Alt text][logo]
-------------------------------------
[回到頂部(索引表格)](#索引表格)

#### 隱藏參考列表
[1]: http://www.google.com   "Google"
[2]: https://markdown.tw/#html  
[3]: https://guides.github.com/features/mastering-markdown/     "Basic writing and formatting syntax"                          
[4]: https://www.markdownguide.org/basic-syntax/                "Markdown Guide"
[logo]: https://www.google.com.tw/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png   

