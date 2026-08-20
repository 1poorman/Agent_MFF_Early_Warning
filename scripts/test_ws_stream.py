"""验证 WebSocket 实时流与 L1/L2/L3 分级预警。"""

import asyncio
import json

import websockets


async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/stream") as ws:
        await ws.send(json.dumps({"fault": "pipe_leak", "speed": 60,
                                  "duration": 350, "fault_start": 150}))
        n = 0
        l1 = 0
        l2max = 0.0
        l3 = None
        async for msg in ws:
            d = json.loads(msg)
            if d.get("done"):
                break
            if "metrics" in d:
                n += 1
                l1 += len(d.get("l1") or [])
                l2max = max(l2max, d["l2"].get("anomaly_score", 0))
                if d.get("l3"):
                    l3 = d["l3"]
        print(f"收到 {n} 条实时数据")
        print(f"L1 预警 {l1} 条")
        print(f"L2 异常分峰值 {l2max:.2f}")
        if l3:
            print(f"L3 诊断: {l3['root_cause']} 置信度 {l3['confidence']}")
        else:
            print("L3 诊断: 无")


if __name__ == "__main__":
    asyncio.run(main())
