#### 部門+區域
``` typescript
import { Observable, from, map, filter, take } from 'rxjs';

const $1 = from([
  { did: '007F01', dnm: '一課', aid: 'A01', anm: ''  },
  { did: '007F02', dnm: '二課', aid: 'A02', anm: ''  },
  { did: '007F03', dnm: '三課', aid: 'A03', anm: '' },
]);
const $2 = from([
  { aid: 'A01', anm: 'TW' },
  { aid: 'A02', anm: 'AZ' },
  { aid: 'A03', anm: 'JP' },
]);

$1.pipe(
  map((x) => {
    let nm = '';
    $2.pipe(
      filter((a) => a.aid === x.aid),
      take(1)
    ).subscribe((a) => {
      nm = a.anm;
    });
    return { ...x, anm: nm };
  })
).subscribe((x) => {
  console.log(x);
});
```
