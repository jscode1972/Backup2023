學習列表
----------------------------
- [Rxjs官網][rxjs]
- [learnRxjs][learn]
- [彈珠圖][rxmarbles]
- [stackblitz][]
- [線上測試][playcode] (8行限制)
- [RxJS筆記][pjchender] (必看!)
```
const $1 = from([
  {did: "007F01", dnm: "一課", aid: "A01" },
  {did: "007F02", dnm: "二課", aid: "A02" },
  {did: "007F03", dnm: "三課", aid: "A01" }
]); 
const $2 = from([
  {aid: "A01", anm:"TW" },
  {aid: "A02", anm:"AZ" }
]); 
```

## Creation Operators
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

## Join Creation Operators
These are Observable creation operators that also have join functionality -- emitting values of multiple source Observables.
- [ ] combineLatest (結合AB最後一個)
- [ ] concat (A+B串接)
- [ ] forkJoin
- [ ] merge (A+B照時序串接)
- [ ] partition
- [ ] race
- [x] zip (AB按序一對一合併)

## Transformation Operators
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

## Filtering Operators
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

## Join Operators
Also see the Join Creation Operators section above.
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

## Error Handling Operators
- [ ] catchError
- [ ] retry
- [ ] retryWhen

## Utility Operators
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

## Conditional and Boolean Operators
- [ ] defaultIfEmpty
- [ ] every
- [ ] find
- [ ] findIndex
- [ ] isEmpty

## Mathematical and Aggregate Operators
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
