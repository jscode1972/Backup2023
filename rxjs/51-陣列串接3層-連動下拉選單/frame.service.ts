import { Injectable } from '@angular/core';
import { Observable, of, delay, from, map, toArray, filter, take, tap, switchMap, mergeMap } from 'rxjs';
import { Area, Nation, City, Full, AREAS, NATIONS, CITYS } from './models';

@Injectable({
  providedIn: 'root'
})
export class FrameService {

  constructor() { }

  getAreas(): Observable<Area[]> {
    return of(AREAS).pipe(delay(1000));
  }

  getCountry(): Promise<Nation[]> {
    return new Promise((resolve, reject) => {
      // 只能配合 Promise?
      setTimeout(() => {
        resolve(NATIONS);
      }, 300);
    });
  }

  getCity(): Observable<City[]> {
    return of(CITYS).pipe(delay(700));
  }

  getFull() : Observable<any> {
    return from(CITYS).pipe(
      mergeMap((city) => {
        // 第二層
        return from(NATIONS).pipe(
          filter((nation) => nation.nid === city.nid),
          take(1),
          switchMap((n) => {
            // 第三層
            return from(AREAS).pipe(
              filter((x) => x.aid === n.aid),
              take(1),
              map((a:Area) => ({ nid: n.nid, nation: n.nation, aid: a.aid, area: a.area }))
            );
          }),
          // 合併 國家,洲別
          map((ref) => ({ ...city, ...ref })) 
        );
      }),
      toArray()
    );
  }
}
