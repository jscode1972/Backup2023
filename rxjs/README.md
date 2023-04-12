學習列表
----------------------------
- [Rxjs官網][rxjs]
- [learnRxjs][learn]
- [彈珠圖][rxmarbles]
- [stackblitz][]
- [線上測試][playcode] (8行限制)
- [RxJS筆記][pjchender] (必看!)

## 學習主題
- 三個步驟 Create/Combine/Listen(建立/組合/監聽)
- 三個方法 next/error/complete
- 操作符 pipe(filter+map) 

## Operators
- [建立類](#建立類) Creation Operators
- [組合建立](#組合建立) Join Creation Operators
- [轉換類](#轉換類) Transformation Operators
- [過濾類](#過濾類) Filtering Operators
- [組合類](#組合類)Join Operators
- [多播類](#多播類) Multicasting Operators 
- [錯誤處理](#錯誤處理) Error Handling Operators
- [工具類](#工具類) Utility Operators
- [條件布林](#條件布林) Conditional and Boolean Operators
- [數學聚合](#數學聚合) Mathematical and Aggregate Operators

#### 建立類
Creation Operators
- [ ] ajax
- [ ] bindCallback
- [ ] bindNodeCallback
- [ ] defer
- [ ] empty
- [x] from (陣列)
- [ ] fromEvent
- [ ] fromEventPattern
- [ ] generate
- [x] interval (按指定毫秒間隔發出,無數次)
- [x] of (物件..)
- [ ] range
- [ ] throwError
- [x] timer (按指定毫秒再發出,僅一次)
- [ ] iif

#### 組合建立
Join Creation Operators
These are Observable creation operators that also have join functionality -- emitting values of multiple source Observables.
- [ ] combineLatest (結合AB最後一個)
- [ ] concat (A+B串接)
- [ ] forkJoin
- [ ] merge (A+B照時序串接)
- [ ] partition
- [ ] race
- [x] zip (AB按序一對一合併)

#### 轉換類
Transformation Operators
- [ ] buffer
- [ ] bufferCount
- [ ] bufferTime
- [ ] bufferToggle
- [ ] bufferWhen
- [ ] concatMap
- [ ] concatMapTo
- [ ] exhaust
- [ ] exhaustMap
- [ ] expand
- [ ] groupBy
- [ ] map
- [ ] mapTo
- [ ] mergeMap
- [ ] mergeMapTo
- [ ] mergeScan
- [ ] pairwise
- [ ] partition
- [ ] pluck
- [ ] scan
- [ ] switchScan
- [ ] switchMap
- [ ] switchMapTo
- [ ] window
- [ ] windowCount
- [ ] windowTime
- [ ] windowToggle
- [ ] windowWhen

#### 過濾類
Filtering Operators
- [ ] audit
- [ ] auditTime
- [ ] debounce
- [ ] debounceTime
- [ ] distinct
- [ ] distinctUntilChanged
- [ ] distinctUntilKeyChanged
- [ ] elementAt
- [ ] filter
- [ ] first
- [ ] ignoreElements
- [ ] last
- [ ] sample
- [ ] sampleTime
- [ ] single
- [ ] skip
- [ ] skipLast
- [ ] skipUntil
- [ ] skipWhile
- [ ] take
- [ ] takeLast
- [ ] takeUntil
- [ ] takeWhile
- [ ] throttle
- [ ] throttleTime

#### 組合類
Join Operators
- [ ] combineLatestAll
- [ ] concatAll
- [ ] exhaustAll
- [ ] mergeAll
- [ ] switchAll
- [x] startWith (強迫加入插入第一個元素pipe)
- [ ] withLatestFrom
- [ ] Multicasting Operators
- [ ] multicast
- [ ] publish
- [ ] publishBehavior
- [ ] publishLast
- [ ] publishReplay
- [ ] share

#### 多播類
Multicasting Operators
- [ ] 缺漏

#### 錯誤處理
Error Handling Operators
- [ ] catchError
- [ ] retry
- [ ] retryWhen

#### 工具類
Utility Operators
- [ ] tap
- [ ] delay
- [ ] delayWhen
- [ ] dematerialize
- [ ] materialize
- [ ] observeOn
- [ ] subscribeOn
- [ ] timeInterval
- [ ] timestamp
- [ ] timeout
- [ ] timeoutWith
- [ ] toArray

#### 條件布林
Conditional and Boolean Operators
- [ ] defaultIfEmpty
- [ ] every
- [ ] find
- [ ] findIndex
- [ ] isEmpty

#### 數學聚合
Mathematical and Aggregate Operators
- [ ] count
- [ ] max
- [ ] min
- [ ] reduce


[rxjs]: https://rxjs.dev/guide/operators "官網"
[learn]: https://www.learnrxjs.io/ "學習"
[rxmarbles]: https://rxmarbles.com/ "彈珠圖"
[playcode]: https://playcode.io/rxjs "測試語法(8行)"
[stackblitz]: https://stackblitz.com/edit/rxjs-m7wtmv?devtoolsheight=60&file=index.ts
[pjchender]: https://pjchender.dev/npm/npm-rx-js/ "RxJS筆記"
