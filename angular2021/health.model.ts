export interface HealthLog {
    date: Date,       // 日期
    temp: Number,     // 溫度
    // 體重計
    weight: Number,     // 體重
    BMI: Number,        // BMI
    bodyFat: Number,    // 體脂肪
    metabolic: Number,  // 代謝
    // 心跳血壓
    BPM: Number,        // 心跳(beat per minute)
    SP: Number,         // 收縮壓(Systolic Pressure)，簡稱SP。
    DP: Number,         // 舒張壓(Diastolic Pressure)，簡稱DP。
    // 運動紀錄
    distance: Number   // 跑步距離
}
