#### 洲際/國家/城市
``` typescript
import { Observable, from, map, filter, take } from 'rxjs';

export const AREAS : Area[] = [
    { aid: 'Asia', area: '亞洲(5)' },
    { aid: 'America', area: '美洲(2)' },
    { aid: 'Europe', area: '歐洲(3)' },
];
  
export const NATIONS : Nation[] = [
  { nid: 'TW', country: '台灣(5)', aid: 'Asia' },
  { nid: 'JP', country: '日本(4)', aid: 'Asia' },
  { nid: 'KR', country: '韓國', aid: 'Asia' },
  { nid: 'HK', country: '香港', aid: 'Asia' },
  { nid: 'US', country: '美國(4)', aid: 'America' },
  { nid: 'GR', country: '德國', aid: 'Europe' },
  { nid: 'IT', country: '義大利', aid: 'Europe' },
];

export const CITYS : City[] = [
    // 5 個
    { cid: 'Taipei',    city: '台北', nid: 'TW' },
    { cid: 'Tainan',    city: '台南', nid: 'TW' },
    { cid: 'Kaohsiung', city: '高雄', nid: 'TW' },
    { cid: 'Tokyo',   city: '東京',   nid: 'JP' },
    { cid: 'Hokaido', city: '北海道', nid: 'JP' },
    { cid: 'Okinawa', city: '沖繩',   nid: 'JP' },
    { cid: 'HongKong',   city: '香港',   nid: 'JP' },
    { cid: 'NewYork',   city: '紐約',   nid: 'US' },    
    { cid: 'Texas',   city: '德州',   nid: 'US' },   
    { cid: 'Frankfurt',   city: '法蘭克福',   nid: 'GR' },  
    { cid: 'Florence',   city: '佛羅倫斯',   nid: 'IT' }
];

```

#### 程式碼 
```typescript
// AI 幫忙修正
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
              map((a) => ({ country: n.country, area: a.area }))
            );
          }),
          // 合併 國家,洲別
          map((ref) => ({ ...city, ...ref })) 
        );
      }),
      toArray()
    );
```
