import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, of } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class HttpService {
   // 資料來源 URL to web api
  //private sUrl = 'https://cmd-20200913.appspot.com/users';  // 外部網站ＡＰＩ

  /* https://medium.com/zero-equals-false/using-cors-in-express-cac7e29b005b
  const httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json' // 沒有也沒差
    })
  };*/

  constructor(private http: HttpClient) { }

  // 非同步寫法 => 觀察者模式, 呼叫端需搭配 subscribe
  get(url : string): Observable<any> {
    // TODO: send the message _after_ fetching the heroes (此處放真實抓資料程式碼)
    //this.messageService.add('HeroService: fetched heroes');
    //return of(HEROES);
    return this.http.get<any>(url);
  }

  // 非同步寫法 => 觀察者模式, 呼叫端需搭配 subscribe
  post(url : string, body : any): Observable<any> {
    //let body = { a: 3 };
    return this.http.post<any>(url, body);
  }

  // 非同步寫法 => sync/await, 呼叫端需搭配 then..catch..
  async getAsync(url : string) {
    // 可試試 拋出錯誤 throw
    //throw new Error('自訂錯誤');         // catch 
    return await this.http.get<any>(url); // then
  }

  // 非同步寫法 => async/await, 呼叫端需搭配 then..catch..
  async postAsync(url : string, body : any) {
    // 可試試 拋出錯誤 throw
    //throw new Error('自訂錯誤');                // catch 
    return await this.http.post<any>(url, body); // then
  }
}
