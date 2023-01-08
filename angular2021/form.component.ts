import { Component, OnInit } from '@angular/core';
//import { Hero } from '../../models/demo/Hero';
import { HttpService } from '../../services/common/http.service';

class Hero {
  constructor(
    public id: number,
    public name: string,
    public email: string,
    //public alterEgo?: string
  ) {  }
}

@Component({
  selector: 'app-form',
  templateUrl: './form.component.html',
  styleUrls: ['./form.component.css']
})
export class FormComponent implements OnInit {

  hero= new Hero(221, 'Dr IQ', 'abc@gmail.com');
  sUrl1 = 'https://cmd-20200913.appspot.com/users';
  sUrl2 = 'http://localhost:3000/users'; // 無 ssl
  values = '';
  constructor(private http : HttpService) { }

  ngOnInit(): void {
    //console.log(this.model);
  }

  onClick() {
    console.log(this.hero);
    this.http.post(this.sUrl2, this.hero ).subscribe(s => {
      console.log('onClick', s);
      //this.title2 = s.name;
    });
  }

  onKey(event: any) { // without type info
    console.log(event);
    this.values += event.target.value + ' | ';
  }
}
