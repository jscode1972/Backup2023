import { Component, Input, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, FormControl, Validators } from '@angular/forms';
import { Subject, zip, switchMap, Observable, of, from, tap, filter, map, distinct, toArray, forkJoin } from 'rxjs';
import { Area, Nation, City, Full } from '../models';
import { FrameService } from '../frame.service';

@Component({
  selector: 'app-select-frame',
  templateUrl: './select-frame.component.html',
  styleUrls: ['./select-frame.component.css']
})
export class SelectFrameComponent implements OnInit {
  @Input() aid : string = '';
  @Input() allowAll : boolean = false;
  //
  areas!: Area[];     // 洲
  nations!: Nation[]; // 國家
  cities!: City[];    // 城市
  fulls!: Full[];     // 合併三層
  alias!: string[];   // 模擬 ClassMain, ClassSub
  // 下拉選單推送物件
  areaSource$ = new Subject<string>();
  nationSource$ = new Subject<string>();
  citySource$ = new Subject<string>();
  // 查詢條件
  form : FormGroup =  this.fb.group({
    aid: this.fb.control('', [Validators.required]),
    nation: this.fb.control('', [Validators.required]),
    cid: this.fb.control('', [Validators.required]),
  });

  constructor(public fb: FormBuilder,
              private svc: FrameService) {
  }

  ngOnInit() {
    this.bindEvent();
    this.zipArray();
  }

  zipArray() {
    // 需三個一起才完成  原本要呼叫三次 (需全部完成再開始)
    //forkJoin([ ]) // 當各只有一個 emit, 等效
    zip( 
      this.svc.getAreas(),
      this.svc.getCountry(),
      this.svc.getCity(),
      this.svc.getFull()
    ).subscribe((arr) => {
      this.areas = arr[0];   // continent
      this.nations = arr[1]; // nation 
      this.cities = arr[2];  // city
      this.fulls = arr[3];  
      // 預設最上層下拉選項
      let aid = this.aid ? this.aid : this.areas[0].aid;
      this.form.patchValue({ aid: aid}); // 這一行很重要
    })
  }

  bindEvent() {
    // 洲別選項推播
    this.areaSource$
      .pipe( switchMap( aid => this.prepareNations(aid) ) )
      .subscribe( (val) => {
        // do things
      });
    // 國家選項推播
    this.nationSource$
      .pipe( switchMap( nation => this.prepareCity(nation) ) )
      .subscribe( (val) => {
        // do things
      });
  }

  prepareNations(aid : string) : Observable<string|null> {
    // 清除下層設定, 這一行很重要
    this.form.patchValue( { nation: null, cid: null });
    // 回傳國家下拉選單 observable
    return from(this.fulls).pipe(
      filter((area) => (area.aid === aid) || ('ALL' === aid) ),
      map((o) => o.nation),
      distinct(),
      toArray(),
      map((arr) => arr.sort((x,y) => x<y ? -1 : 1 )),
      map((arr) => {
        this.alias = arr;
        let nation =this.allowAll ? 'ALL' : (arr.length>0 ? arr[0] : null );
        return nation;
      }), 
      // 如果只要預設 area, 但不預設下一層 nation, 此行拿掉即可
      tap((nation) => this.form.get("nation")?.setValue(nation) )
    );
  }

  prepareCity(nation : string) : Observable<any> {
    // 清除下層設定, 這一行很重要
    this.form.patchValue({ cid: null });
    // 回傳城市下拉 observable
    return from(this.fulls).pipe(
      filter((arr) => arr.aid === this.form.get("aid")?.value ),
      filter((arr) => (arr.nation === nation) || ('ALL' === nation) ),
      toArray(),
      map((arr) => arr.sort((x,y) => x.city<y.city ? -1 : 1 )),
      map((arr) => {
        this.cities = arr;
        let cid = this.allowAll ? 'ALL' : (arr.length>0 ? arr[0].cid : null );
        return cid;
      }),
      tap((cid) => this.form.get("cid")?.setValue(cid) ) 
    );
  }
}
