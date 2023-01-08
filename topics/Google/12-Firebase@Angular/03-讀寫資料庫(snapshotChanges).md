## 讀寫資料庫 (snapshotChanges)

#### 本節講解重點
* [參考來源官網](https://github.com/angular/angularfire/blob/master/docs/firestore/collections.md)
* snapshotChanges()
  * 它是 Observable of data as a synchronized array of **DocumentChangeAction[]**.
  * Why would you use it? 
    * 當你需要複雜資料以方便操控資料(保留元數據, id..)
  * When would you not use it? 
    * 原文不太懂
  * Best practices
* [DocumentChangeAction](#DocumentChangeAction) 型別
* [使用範例](#使用範例) (snapshotChanges)


#### DocumentChangeAction
```typescript
interface DocumentChangeAction {
  //'added' | 'modified' | 'removed';
  type: DocumentChangeType;
  payload: DocumentChange;
}

interface DocumentChange {
  type: DocumentChangeType;
  doc: DocumentSnapshot;
  oldIndex: number;
  newIndex: number;
}

interface DocumentSnapshot {
  exists: boolean;
  ref: DocumentReference;
  id: string;
  metadata: SnapshotMetadata;
  data(): DocumentData;
  get(fieldPath: string): any;
}
```

#### 使用範例
```html
<ul>
  <li *ngFor="let shirt of shirts | async">
    {{ shirt.name }} is {{ shirt.price }}
  </li>
</ul>
```
```typescript
import { Component, OnInit } from '@angular/core';
import { AngularFirestore, AngularFirestoreCollection } from '@angular/fire/firestore';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

export interface Shirt { name: string; price: number; }      // 不同前例
export interface ShirtId extends Shirt { id: string; }       // 不同前例

@Component({ ...  })

export class LinkComponent implements OnInit {
  private shirtCollection: AngularFirestoreCollection<Shirt>;
  shirts: Observable<ShirtId[]>;
  constructor(private readonly afs: AngularFirestore) {
    this.shirtCollection = afs.collection<Shirt>('shirts');
    this.shirts = this.shirtCollection.snapshotChanges().pipe(
      map(actions => actions.map(a => {
        const data = a.payload.doc.data() as Shirt;
        const id = a.payload.doc.id;
        return { id, ...data };
      }))
    );
    // .snapshotChanges() returns a DocumentChangeAction[], which contains
    // a lot of information about "what happened" with each change. If you want to
    // get the data and the id use the map operator.
  }

}

```
