## 重點步驟 (此處有點複雜, 腦袋轉一下)
* 官方網址 https://angular.io/tutorial/toh-pt4
* 建立訊息服務 MessageService 
* 建立訊息組件 MessagesComponent (消費者, 顯示訊息)
  * 注入一個服務 (**public** MessageService) 
  * public 是因為 template 要參考
* 修改組件 Heroes (生產者, 產生訊息)
  * 注入兩個服務 (HeroService & MessageService)
  * 本節重點在於使用後者訊息服務 (上一節是使用服務 => 抓取資料)
* 修改服務 HeroService (生產者, 產生訊息)
  * 注入一個服務 (MessageService)

## 創建服務 & 組件 (Service & Component)
* 創建服務 MessageService (處理訊息) => **可用來多重注入**
  ```
  $ ng generate service message
  ```
* 創建組件 MessagesComponent (顯示訊息)
  ```
  $ ng generate component messages
  ```
 
## 程式碼範例
#### Heroes 組件 (透過注入, 使用兩個服務 1.抓取資料 2.顯示訊息)
* heroes.component.html (不動)
  ```
  // 略, 不動
  ```
* heroes.component.ts (修改)
  * 只列出異動部分
  ```typescript
  import { MessageService } from '../message.service';
  // 略
  export class HeroesComponent implements OnInit {
    // 略
    // 修改部分
    constructor(private heroService: HeroService,          // <-- 重點, 注入服務物件
                private messageService: MessageService) {  // <-- 重點, 注入服務物件, 新增部分
    }
    // 略
    // 修改部分
    onSelect(hero: Hero): void {
      this.selectedHero = hero;
      this.messageService.add(`HeroesComponent: Selected hero id=${hero.id}`); // 新增部分
    }
  }
  ```
  
#### Messages 組件 (透過注入 MessageService, 顯示訊息)
* messages.component.html (新增)
```html
<div *ngIf="messageService.messages.length">

  <h2>Messages</h2>
  <button class="clear"
          (click)="messageService.clear()">clear</button>
  <div *ngFor='let message of messageService.messages'> {{message}} </div>

</div>
```
* messages.component.ts (新增)
```typescript
import { Component, OnInit } from '@angular/core';
import { MessageService } from '../message.service';

@Component({
  selector: 'app-messages',
  templateUrl: './messages.component.html',
  styleUrls: ['./messages.component.css']
})
export class MessagesComponent implements OnInit {

  constructor(public messageService: MessageService) { } // <-- public 是因為 template 要參考)

  ngOnInit(): void {}
}
```

#### HeroService 服務 (抓取資料 => 留下紀錄)
* src/app/**hero**.service.ts (修改)
```typescript
import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { Hero } from './hero';
import { HEROES } from './mock-heroes';
import { MessageService } from './message.service';
// 略
export class HeroService {
  // 修改部分
  constructor(private messageService: MessageService) { } // <-- 重點, 注入服務物件, 新增部分
  // 略...
  // 非同步寫法 => 正解!
  getHeroes(): Observable<Hero[]> {
    // TODO: send the message _after_ fetching the heroes (此處放真實抓資料程式碼)
    this.messageService.add('HeroService: fetched heroes');   // <-- 新增部分
    return of(HEROES);
  }
}
```

#### MessageService 服務 (處理訊息)
* src/app/**message**.service.ts (新增)
```typescript
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class MessageService {

  messages: string[] = [];

  constructor() { }

  add(message: string) {
    this.messages.push(message);
  }

  clear() {
    this.messages = [];
  }
}
```
