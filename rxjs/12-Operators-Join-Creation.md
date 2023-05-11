## 索引表格
打通Rxjs任督二脈 (每個主題很多範例
- [ ] [combineLatest](#combineLatest)
- [ ] concat 
- [ ] forkJoin (需觸發 complete, 適合 http)
- [ ] merge
- [ ] partition
- [ ] race
- [ ] zip (一對一觸發)

## 範例網站
- [圖示範例][digitalocean] (forkjoin-zip-combinelatest-withlatestfrom)
- [Day29-基礎篇][Day29] 
 
#### combineLatest
觸發結合每一組**最後一個新元素**  
```typescript
// 在剛剛討論 switchMap 時，我們的呼叫是有順序的，而當沒有順序時，我們可能會希望平行的處理 observable，  
// 並將所有 observable 有資料後才進行後續處理，這時候就可以使用 combineLatest 來同時取得資料，不會有順序問題！
const posts$ = this.httpClient.get('.../posts');
const tags$ = this.httpClient.get('.../tags');
this.data$ = combineLatest(posts$, tags$).pipe(
  map(([posts, tags]) => ({posts: posts, tags: tags}))
)

// 我們也可以整合畫面上各種事件最終得到結果，例如一個包含搜尋、排序和分頁的資料，
// 我們可以將搜尋、排序和分頁都設計成單一個 observable，在使用 combineLatest 產生搜尋結果，如下：
this.products$ = combineLatest(
  this.filterChange$,
  this.sortChange$,
  this.pageChange$
)
.pipe(
  exhaustMap(([keyword, sort, page]) =>
    this.httpClient
      .get(`.../products/?keyword=${keyword}&sort=${sort}&page=${page}`)
  )
);
```
[回到頂部(索引表格)](#索引表格)

#### concat
可接受的參數:
```typescript

```
[回到頂部(索引表格)](#索引表格)


#### forkJoin
觸發時機: 適合http, 或是送出 complete 之訂閱
```typescript
// forkJoin 與 combineLatest 類似，差別在於 combineLatest 在 RxJS 整個資料流有資料變更時都會發生，
// 而 forkJoin 會在所有 observable 都完成(complete)後，才會取得最終的結果，所以對於 Http Request 的整合，
// 我們可以直接使用 forkJoin 因為 Http Request 只會發生一次，然後就完成了！
const posts$ = this.httpClient.get('.../posts');
const tags$ = this.httpClient.get('.../tags');
this.data$ = forkJoin(posts$, tags$).pipe(
  map(([posts, tags]) => ({posts: posts, tags: tags}))
)
```
[回到頂部(索引表格)](#索引表格)


#### merge
可接受的參數:
```typescript

```
[回到頂部(索引表格)](#索引表格)


#### partition
可接受的參數:
```typescript

```
[回到頂部(索引表格)](#索引表格)


#### race
可接受的參數:
```typescript

```
[回到頂部(索引表格)](#索引表格)


#### zip
一對一觸發
```typescript

```
[回到頂部(索引表格)](#索引表格)


[digitalocean]:https://www.digitalocean.com/community/tutorials/rxjs-operators-forkjoin-zip-combinelatest-withlatestfrom#forkjoin
[Day29]: https://ithelp.ithome.com.tw/m/articles/10209779 "Day29 在 Angular 中應用 RxJS 的 operators (1) - 基礎篇"
