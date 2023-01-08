## 讀寫資料庫 (stateChanges)

#### 本節講解重點
* [參考來源官網](https://github.com/angular/angularfire/blob/master/docs/firestore/collections.md)
* stateChanges()
  * Returns an Observable of the most recent changes as a DocumentChangeAction[]
  * Why would you use it? 
    * The above methods return a synchronized array sorted in query order. stateChanges() emits changes as they occur rather than syncing the query order. This works well for ngrx integrations as you can build your own data structure in your reducer methods.
  * When would you not use it? 
    * When you just need a list of data. This is a more advanced usage of AngularFirestore.
  * Best practices
* [使用範例](#使用範例) (snapshotChanges)


#### 使用範例
* 有空再補
```html

```
```typescript


```
