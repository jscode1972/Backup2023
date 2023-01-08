import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { catchError, map, tap } from 'rxjs/operators';
import { Opendata } from './opendata';
import { Nasadata } from './nasadata';

@Injectable({
  providedIn: 'root'
})

export class DataService {
  // API 網址
  private nasaUrl = 'https://api.nasa.gov/planetary/apod?api_key=NNKOjkoul8n1CH18TWA9gwngW1s1SmjESPjNoUFo';
  private govUrl = 'https://cloud.culture.tw/frontsite/trans/emapOpenDataAction.do?method=exportEmapJsonByMainType&mainType=10';
  private awsObjUrl = 'http://iquake.org:8080/data';
  private awsFsUrl = 'http://iquake.org:8080/json';
  // 參考網址
  // https://semlinker.com/ng-jsonp-detail/
  // https://stackoverflow.com/questions/36289495/how-to-make-a-simple-jsonp-asynchronous-request-in-angular-2/47643846#47643846

  constructor(private http: HttpClient) {
    console.log('DataService is created.');
  }

  private log(message: string) {
    //this.messageService.add(`HeroService: ${message}`);
  }

  // NASA 無 CORS 問題 (看表頭就知道,
  // 表頭: Access-Control-Allow-Origin: *)
  getNasa() : Observable<Nasadata> {
    return this.http.get<Nasadata>(this.nasaUrl);
  };
  // 政府平台 API 有 CORS 問題 (看 API 表頭就知道, Access-Control-Allow-Origin: *)
  getGov() : Observable<Opendata[]> {
      return this.http.get<Opendata[]>(this.govUrl)
        .pipe(tap(_ => this.log('fetched GovJson')),
              catchError(this.handleError('getGov', []))
      );
  }
  // 亞馬遜 物件 => JSONP
  getAwsObject(): Observable<Opendata[]>  {
    // Pass the key for your callback (in this case 'callback')
    // as the second argument to the jsonp method
    return this.http.jsonp<Opendata[]>(this.awsObjUrl, 'callback');
  }
  // 亞馬遜 檔案 => JSONP
  getAwsFile(): Observable<Opendata[]>  {
    // Pass the key for your callback (in this case 'callback')
    // as the second argument to the jsonp method
    return this.http.jsonp<Opendata[]>(this.awsFsUrl, 'callback');
  }
/**
 * Handle Http operation that failed.
 * Let the app continue.
 * @param operation - name of the operation that failed
 * @param result - optional value to return as the observable result
 */
private handleError<T> (operation = 'operation', result?: T) {
  return (error: any): Observable<T> => {
    // TODO: send the error to remote logging infrastructure
    console.error(error); // log to console instead
    // TODO: better job of transforming error for user consumption
    this.log(`${operation} failed: ${error.message}`);
    // Let the app keep running by returning an empty result.
    return of(result as T);
  };
}
}
