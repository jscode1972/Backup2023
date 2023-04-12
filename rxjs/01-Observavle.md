## 索引
- Observer
- Subscription
- Observavle
- Subject
- ReplaySubject
- BehaviorSubject

#### Observer
觀察者
```typescript
// 2023/04/12 剛好用到
const observer = {
  next: (data) => { console.log(data) },
  error: (err) => { console.log(err) },
  complete: () => { console.log('done') }
}

// 1.完整寫法
source.subscribe(observer);

// 2.只處理 next
source.subscribe({
  next: (data) => console.log(data);
});

// 3.直接傳入一個方法當作參數 
source.subscribe(
  (data) => console.log(data) // 如果沒有特別處理錯誤,都會這樣寫
);
```

#### Subscription
訂閱/退訂
```typescript
//訂閱
const subscription : Subscription = source.subscribe( (data) => console.log(data) );
// 發送資料
source.next(1);
source.next(2);
// 退訂
subscription.unsubscribe();
// 此時應該收不到資料
source.next(2);
source.next(3); 
// 觀察是否已退訂
console.log(subscription.closed); 
```

#### Observavle
被觀察者


#### Subject
一般
```typescript
count = 0;
counter$ = new Subject();
counter$.subscribe( 
  (data) => console.log(data) 
);
counter$.next(count++);
```

#### ReplaySubject
重播

#### BehaviorSubject

