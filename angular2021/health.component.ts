import { Component, OnInit } from '@angular/core';
import { HttpService } from '../../services/common/http.service';

/*class Health {
  constructor(
    //public date: Date,
    public weight: Number,
    public BMI: Number,
    public temp: Number,
    //public alterEgo?: string
  ) {  }
}*/

interface Health {
  weight: Number,
  BMI: Number,
  temp: Number,
}

@Component({
  selector: 'app-health',
  templateUrl: './health.component.html',
  styleUrls: ['./health.component.css']
})
export class HealthComponent implements OnInit {

  //health = new Health(68, 22, 33);
  health = {weight:39, BMI:22, temp:29};
  sUrl1 = 'https://cmd-20200913.appspot.com/users';
  sUrl2 = 'http://localhost:3000/users'; // 無 ssl

  constructor(private http : HttpService) {

   }

  ngOnInit(): void { }

  onSubmit(){
    console.log(this.health);
    this.http.post(this.sUrl1, this.health ).subscribe(s => {
      console.log('onClick', s);
      //this.title2 = s.name;
    });
  }

}
