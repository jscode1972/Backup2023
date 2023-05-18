#### 先抓學科類別最新編碼, 再寫入學科項目 switchMap
``` typescript
// 先抓學科類別最新編碼, 再寫入學科項目
this.quizaService.getCatgNo(sid)
  .pipe(
    map( (resp) => { // 取得封裝 resp.data => 類別編碼 eg.'03'
      // 補齊表單內容
      this.form.patchValue({ catgNo : resp.data, createUser : 'xxx' });
      return this.form.value;
    }), 
    switchMap(
      // 送出學科內容
      (form) => this.quizaService.addCatg(form)
    )
  )
  .subsrvibe(
    x => console.log();  
  )
```
