## 索引表格
打通Rxjs任督二脈 (每個主題很多範例)
- [ ] ajax
- [ ] bindCallback
- [ ] bindNodeCallback
- [ ] defer
- [ ] empty
- [ ] [from](#from) (陣列/可迭代的物件(iterable)/Promise/其他observable)
- [ ] fromEvent
- [ ] fromEventPattern
- [ ] generate
- [x] [interval](#interval) (按指定毫秒間隔發出,無數次)
- [x] [of](#of) (傳入的參數)
- [ ] range
- [ ] throwError
- [x] [timer](#timer) 按指定毫秒再發出,僅一次
- [ ] iif

 
#### from
可接受的參數: 陣列,可迭代的物件(iterable),Promise,其他observable
```typescipt
from([1,2,3,4])
  .subscribe(data => {
    console.log(data);
  })
```
[回到頂部(索引表格)](#索引表格)

#### interval
```typescipt
```
[回到頂部(索引表格)](#索引表格)

#### of
可接受的參數: 傳進去的參數
```typescipt
of(1,2,3,4)
  .subscribe(data => {
    console.log(data);
  })
注意 => of([1,2,3,4]) 是傳出一個陣列
```
[回到頂部(索引表格)](#索引表格)

#### timer
```typescipt
```
[回到頂部(索引表格)](#索引表格)
