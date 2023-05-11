## 索引表格
打通Rxjs任督二脈 (每個主題很多範例
- [ ] [concatMap](#concatMap)  (多合一合併,接續)
- [ ] exhaustMap 
- [ ] groupBy
- [ ] map
- [ ] [mergeMap](#mergeMap) (多合一合併,按時序)
- [ ] pairwise 
- [ ] partition
- [ ] scan 
- [ ] [switchMap](#switchMap) (資料流轉換資料流) 避免 anti-pattern

## 範例網站
- [圖示範例][xxx] (xxx)
 
#### concatMap
當需要處理多個資料流的事件時，可以使用 concatMap 操作符將這些資料流連接成一個序列，並按照順序進行處理。  
```typescipt
// 例如，當需要從多個資料源中讀取資料時，可以使用 concatMap 將這些資料源連接成一個序列，然後按照順序依次從每個資料源中讀取資料。
```
[回到頂部(索引表格)](#索引表格)


#### exhaustMap
可接受的參數:
```typescipt

```
[回到頂部(索引表格)](#索引表格)

#### groupBy
```typescipt

```
[回到頂部(索引表格)](#索引表格)


#### map
可接受的參數:
```typescipt

```
[回到頂部(索引表格)](#索引表格)


#### mergeMap
當需要處理**多個資料流**的事件時，可以使用 mergeMap 操作符將這些資料流**合併成一個**資料流。  
```typescipt
// 例如，當需要從多個資料源中讀取資料時，可以使用 mergeMap 將這些資料源合併成一個資料流，然後將其訂閱以獲取所有的資料。
```
[回到頂部(索引表格)](#索引表格)


#### pairwise
可接受的參數:
```typescipt

```
[回到頂部(索引表格)](#索引表格)


#### partition
一對一觸發
```typescipt

```
[回到頂部(索引表格)](#索引表格)


#### scan
一對一觸發
```typescipt

```
[回到頂部(索引表格)](#索引表格)

#### switchMap
當需要將一個資料流轉換為另一個資料流時，可以使用 switchMap 操作符。  
例如，當使用者在一個搜尋欄位中輸入關鍵字時，可以使用 switchMap 將輸入的關鍵字轉換為一個 API 請求的資料流，然後使用該資料流來更新搜尋結果。
```typescipt

```
[回到頂部(索引表格)](#索引表格)


[xxx]: https://
