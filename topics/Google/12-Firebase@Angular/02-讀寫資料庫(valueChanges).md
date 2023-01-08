## 讀寫資料庫 (valueChanges)

#### 本節講解重點
* [參考來源官網](https://github.com/angular/angularfire/blob/master/docs/firestore/collections.md)
* AngularFirestoreCollection
  * [簡單寫入](#簡單寫入) interface Item { name: string; }
  * [進階寫入](#進階寫入) interface Item { id: string; name: string; }
* valueChanges({idField?: string}) 
  * 它是 an Observable of data as a synchronized array of **JSON objects**.
  * 什麼時候使用? 
    * => 只要簡單資料(不含元數據), 方便渲染畫面
  * 什麼時候不要用?
    * => 當你需要比陣列更複雜的資料
  * Best practices
    - Use this method to display data on a page. It's simple but effective. Use .snapshotChanges() once your needs become more complex.
* snapshotChanges()
  * 參考下節
  
#### 簡單寫入
```html
<ul>
  <li *ngFor="let item of items | async">
    {{ item.name }}
  </li>
</ul>
<button (click)="addItem({name:'測試'})">add item</button>     // 寫入資料為介面型態
```
```typescript
import { Component, OnInit } from '@angular/core';
import { AngularFirestore, AngularFirestoreCollection } from '@angular/fire/firestore';
import { Observable } from 'rxjs';

export interface Item { name: string; }                        // <-- 宣告介面

@Component({ .... })

export class LinkComponent implements OnInit {

  private itemsCollection: AngularFirestoreCollection<Item>;   // a wrapper around the native Firestore SDK's 
  items: Observable<Item[]>;                                   //      CollectionReference and Query types.

  constructor(private afs: AngularFirestore) {
    this.itemsCollection = afs.collection<Item>('items');
    this.items = this.itemsCollection.valueChanges();
  }
  
  addItem(item: Item) {                                        // 寫入資料為介面型態
    this.itemsCollection.add(item);
  }
}
```

#### 進階寫入
```html
<ul>
  <li *ngFor="let item of items | async">
    {{ item.name }} ( {{ item.id }})
  </li>
</ul>
<button (click)="addItem('測試2')">add item2</button>
```
```typescript
// 略...
export interface Item { id: string; name: string; }               // 不同上例, 加了 id

@Component({  ..  })

export class LinkComponent implements OnInit {

  private itemsCollection: AngularFirestoreCollection<Item>;     // 同上
  items: Observable<Item[]>;                                     // 同上

  constructor(private readonly afs: AngularFirestore) {          // 不同上例, 加了 readonly
    this.itemsCollection = afs.collection<Item>('items');
    this.items = this.itemsCollection.valueChanges();
  }
    // .valueChanges() is simple. It just returns the
    // JSON data without metadata. If you need the
    // doc.id() in the value you must persist it your self
    // or use .snapshotChanges() instead. See the addItem()
    // method below for how to persist the id with
    // valueChanges()
  
  addItem(name: string) {                        // 參數是字串, 非介面
    // Persist a document id
    const id = this.afs.createId();              // 系統建立資料, 取得 id
    const item: Item = { id, name };
    this.itemsCollection.doc(id).set(item);
  }
}
```

