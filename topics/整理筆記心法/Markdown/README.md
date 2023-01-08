## 語法參考連結
  * [guides.github](https://guides.github.com/features/mastering-markdown/)
  * [Markdown文件](https://markdown.tw/#html)
  * [Markdown Guide](https://www.markdownguide.org/basic-syntax/)
  
-----
### 階層大區塊 (Blockquotes) -> 箭頭 >  
As Kanye West said:
> We're living the future so the present is our past.
>> 這裡再縮一次

-----
### 行內小區塊 (Inline code) -> 反引號x1
I think you should use an `<addr>` element here instead.

-----
### 段落大區塊 (Syntax highlighting) -> 反引號x3
```
function fancyAlert(arg) {
  if(arg) {
    $.facebox({div:'#foo'})
  }
}
```

-----
### 區塊程式碼 (Syntax highlighting) -> 反引號x3+語言
```javascript
function fancyAlert(arg) {
  if(arg) {
    $.facebox({div:'#foo'})
  }
}
```

-----
### 工作清單 (Task Lists) -> - [x]
- [x] @mentions, #refs, [links](), **formatting**, and <del>tags</del> supported
- [x] list syntax required (any unordered or ordered list supported)
- [x] this is a complete item
- [ ] this is an incomplete item

-----
### 表格 (Tables)
You can create tables by assembling a list of words and dividing them with hyphens - (for the first row), and then separating each column with a pipe |:

First Header | Second Header
------------ | -------------
Content from cell 1 | Content from cell 2
Content in the first column | Content in the second column

-----
### 強調重點 (Emphasis)
*This text will be italic* (兩個空格可斷行)  
_This will also be italic_

**This text will be bold** (兩個空格可斷行)  
__This will also be bold__

_You **can** combine them_

-----
### 刪節字 (Strikethrough)
Any word wrapped with two tildes (like ~~this~~) will appear crossed out.

-----
### 水平線
---

