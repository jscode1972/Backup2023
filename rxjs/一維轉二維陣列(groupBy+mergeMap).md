```typescript
import { Component, OnInit } from '@angular/core';
import { from, groupBy, mergeMap, toArray, map } from 'rxjs';

interface MyTuple<T> {
  id?: string;
  group: string;
  title?: string;
  url?: string;
  items?: T[]
}
interface Menu extends MyTuple<Menu> {}

const Forms : Menu[] = [
  {id: "01", group: "A", title: "甲" },
  {id: "02", group: "B", title: "乙" },
  {id: "03", group: "A", title: "丙" },
  {id: "04", group: "C", title: "丁" },
];

@Component({
  selector: 'app-rxjs',
  templateUrl: './rxjs.component.html',
  styleUrls: ['./rxjs.component.css']
})
export class RxjsComponentm implements OnInit {

  menus : Menu[] = [];

  ngOnInit() {
    from(Forms) 
    .pipe(
      map( p => ({...p, active: false })),   // 補充欄位 (... 三個連續點是語法)
      groupBy(p => p.group),                 // 聚合主鍵
      mergeMap($g => $g.pipe( toArray() )),  // 合併子項陣列
      map(arr => ({ group:arr[0].group, open: false, items: arr })), // 重組2D物件
      map( (data,index) => {            
        if (index === 0)                     // 特定條件重新給值
          data.open = true;
        return data;
      }),
      toArray()                              // 轉成陣列 (subscribe 就不會一個個觸發)
    )
    .subscribe( o => {
        console.log(o);
        this.menus = o;
    } );
  }
}
```
