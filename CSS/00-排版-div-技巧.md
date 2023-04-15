排版演變歷史 
--------------
排版演變歷史: table -> float -> inline-block & box (flex-box) -> flex & Grid (參考哥 [cxx-flex] )


#### div 置中
```
.wrapper {
  width: 600px;
  background-color: aqua;
  height: 100vh;
  margin: auto;
}
```
 
#### 文繞圖+清除
```
// 前者靠邊
float: left;  (right)
// 下個復原
clear: both;
```

#### Grid 磚牆
```
grid-temp-col: repeat(3, minmax(240px,1fr));
grid-temp-col: repeat(auto-fit, minmax(240px,1fr)); // 適合圖片
gap: 20px;

.big-box {
  grid-column: 1/3
  grid-row: 1/3
}

@media (max-width: 600px)
  .big-box {
    grid-column: auto;
    grid-row: auto;
  }
  
  .big-box img {
    height: 100%;
  }
```



[cxx-flex]: https://www.youtube.com/watch?v=_nCBQ6AIzDU "玩轉 CSS FLEX (金魚哥)"
