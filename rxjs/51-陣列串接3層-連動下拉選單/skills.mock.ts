import { Area, Nation, City } from "./skills.model";

export const AREAS : Area[] = [
    { aid: 'Asia', area: '亞洲(5)' },
    { aid: 'America', area: '美洲(2)' },
    { aid: 'Africa', area: '非洲(無)' },
    { aid: 'Europe', area: '歐洲(3)' },
];
  
export const NATIONS : Nation[] = [
  { nid: 'TW', nation: '台灣', aid: 'Asia' },
  { nid: 'JP', nation: '日本', aid: 'Asia' },
  { nid: 'KR', nation: '韓國', aid: 'Asia' },
  { nid: 'KR', nation: '韓國', aid: 'Asia' },
  { nid: 'HK', nation: '香港', aid: 'Asia' },
  { nid: 'BZ', nation: '巴西', aid: 'America' },
  { nid: 'US', nation: '美國', aid: 'America' },
  { nid: 'GR', nation: '德國', aid: 'Europe' },
  { nid: 'FR', nation: '法國', aid: 'Europe' },
  { nid: 'IT', nation: '義大利', aid: 'Europe' },
];

export const CITYS : City[] = [
    // 5 個
    { cid: 'Taipei',    city: '台北', nid: 'TW' },
    { cid: 'Tainan',    city: '台南', nid: 'TW' },
    { cid: 'Kaohsiung', city: '高雄', nid: 'TW' },
    { cid: 'Pingtung',  city: '屏東', nid: 'TW' },
    { cid: 'Huanlian',  city: '花蓮', nid: 'TW' },

    { cid: 'Tokyo',   city: '東京',   nid: 'JP' },
    { cid: 'Hokaido', city: '北海道', nid: 'JP' },
    { cid: 'Okinawa', city: '沖繩',   nid: 'JP' },
    { cid: 'Osaka',   city: '大阪',   nid: 'JP' },

    { cid: 'HongKong',   city: '香港',   nid: 'HK' },

    { cid: 'São Paulo',   city: '聖保羅',   nid: 'BZ' },
    { cid: 'Rio de Janeiro',   city: '里約熱內盧',   nid: 'BZ' },


    { cid: 'NewYork',   city: '紐約',   nid: 'US' },    
    { cid: 'LA',   city: '洛杉磯',   nid: 'US' },   
    { cid: 'Washionton',   city: '華盛頓',   nid: 'US' },   
    { cid: 'Texas',   city: '德州',   nid: 'US' },  
    
    { cid: 'Berlin',   city: '柏林',   nid: 'GR' },  
    { cid: 'Hamburg',   city: '漢堡',   nid: 'GR' },  
    { cid: 'München',   city: '慕尼黑',   nid: 'GR' },  
    { cid: 'Frankfurt',   city: '法蘭克福',   nid: 'GR' },  

    { cid: 'Rome',   city: '羅馬',   nid: 'IT' },  
    { cid: 'Milano',   city: '米蘭',   nid: 'IT' },  
    { cid: 'Florence',   city: '佛羅倫斯',   nid: 'IT' }

];

