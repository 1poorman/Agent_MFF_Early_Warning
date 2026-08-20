# Agent_MFF_Early_Warning

中频炉水冷系统多参数融合预警智能体（第十一届"创客中国"工业智能体大赛）。

## 环境

```bash
conda create -n mff_agent python=3.10 -y
conda activate mff_agent
pip install -r requirements.txt
```

## 数据模拟智能体（simulator/）

基于物理机理模型生成 1Hz 传感器时序数据，各物理量严格满足：

- 炉体热平衡：`C·dT/dt = P·η - h·(T - T_amb)`
- 冷却水热平衡：`T_out = T_in + Q_heat / (c·ṁ)`
- 管网特性：`P = P_static + R·Q²`，`v = Q/A`
- 电气公式：`P = √3·U·I·cosφ`

### 输出字段

| 字段 | 单位 | 说明 |
|---|---|---|
| timestamp | — | 1s 连续递增 |
| inlet_temp / outlet_temp | ℃ | 冷却水进/出水温度 |
| pressure | kPa | 管道压力（正常 150~300） |
| flow_rate | L/s | 冷却水流量（额定 8.0） |
| flow_velocity | m/s | 管道流速（DN65） |
| tank_level | cm | 水箱液位 |
| conductivity | µS/cm | 电导率（水质） |
| cabinet_temp / cabinet_humidity | ℃ / %RH | 电气柜表面温度 / 环境湿度 |
| furnace_temp | ℃ | 炉内温度 |
| electric_power / electric_current | kW / A | 电功率 / 电流 |
| operating_condition | — | 工况：startup/melting/holding/tapping/idle |
| fault_label | — | 注入故障标签（none 为正常） |

### 使用

```bash
# 24h 正常数据
python -m simulator --duration 86400 --out data/simulated/normal_24h.csv

# 注入故障（名称@起始秒:爬升秒:程度0~1）
python -m simulator --duration 21600 \
    --faults "filter_clog@10800:1800:0.9" "pipe_leak@14400:900:0.7" \
    --out data/simulated/fault_demo_6h.csv
```

可选故障：`filter_clog`（过滤器堵塞）、`pump_cavitation`（水泵气蚀）、
`pipe_leak`（管道泄漏，联动湿度上升+液位下降）、`scale_buildup`（水垢缓变）。

输出格式支持 CSV / parquet（`--format parquet`）。

### 自测

```bash
python tests/test_simulator.py
```

覆盖：数据质量（无空值/乱序、1s 连续、1 位小数）、热平衡重建误差、
P-Q² 水力特性、电气公式一致性、故障注入的物理联动。
