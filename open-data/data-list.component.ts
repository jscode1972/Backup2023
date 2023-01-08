import { Component, OnInit } from '@angular/core';
//import { OpenDatas } from '../opendata-mock';
import { Opendata } from '../opendata';
import { Nasadata } from '../nasadata';
import { DataService } from '../data.service';

@Component({
  selector: 'app-data-list',
  templateUrl: './data-list.component.html',
  styleUrls: ['./data-list.component.css']
})

export class DataListComponent implements OnInit {

  nasa : Nasadata;
  jsonGov : Opendata[];
  jsonAwsObj : Opendata[];
  jsonAwsFs : Opendata[];

  constructor(private dataService : DataService) { }

  ngOnInit() {
    this.getNasa();      // OK
    this.getGov();       // 不 OK (需要 Access or JSONP )
    this.getAwsObject(); // OK
    this.getAwsFile();   // OK
  }

  // NASA
  // 開放Access-Control-Allow-Origin: *
  getNasa() : void {
    this.dataService.getNasa()
      .subscribe(nasa => this.nasa = nasa);
  }
  // 原始政府平台 CORS 限制 (無 JSONP 技術)
  // 亦無開放Access-Control-Allow-Origin: *
  getGov() : void {
    this.dataService.getGov()
      .subscribe(data => this.jsonGov = data);
  }
  // 亞馬遜 CORS 限制 (有 JSONP 技術)
  // 字串模式輸出 JSONP
  getAwsObject() : void {
    this.dataService.getAwsObject()
      .subscribe(data => this.jsonAwsObj = data);
  }
  // 亞馬遜 CORS 限制 (有 JSONP 技術)
  // 字串模式輸出 JSONP
  getAwsFile() : void {
    this.dataService.getAwsFile()
      .subscribe(data => this.jsonAwsFs = data.slice(1,15));
  }
}
