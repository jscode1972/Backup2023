### 連動選單 
學到什麼????
- FrameService
  - 將三個上下層關係陣列, 透過鍵值組合打平整併
  - from/mergeMap/filter/switchMap
- select-frame.component.ts/html
  - [x] 等待多個 Observable 到位再動作
  - [x] 重複使用上面資料展開不同選單
  - [x] 指定 @input 預設最上層起始值
  - [x] 自訂 @input 單選/多選(ALL)
  - [x] 可自動連動多重選單 (透過下拉選項推播 Subject)
    - [x] (ngModelChange)="nationSource$.next($event)" 
  - [x] 可組合成一組 form.value 當後端參數 parameter model
  - [x] 過濾資料 -> 下拉選單 -> 指定預設值 -> 支援ALL (一氣呵成)
    - [x] switchMap => 自訂 return function Observable<T> 更有層次, 模組化
    - [x] filter, distinct, toArray
    - [x] sort(x,y) 排序

  
  
