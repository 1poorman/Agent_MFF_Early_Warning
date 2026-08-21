"""时序数据库存储验收：插入/查询/分区自动管理。"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage import TimeSeriesDB


def main():
    db = TimeSeriesDB()
    try:
        # 1. 插入
        from simulator import DataSimulator, SimConfig
        sim = DataSimulator(config=SimConfig(seed=1))
        rows = []
        for t in range(3):
            r = sim._sense(sim.step())
            r["timestamp"] = f"2026-08-21 12:00:0{t}"
            rows.append(r)
        n = db.insert(rows)
        print(f"[PASS] 插入 {n} 条")
        assert n == 3

        # 2. 查询
        df = db.query(start="2026-08-21 00:00:00", end="2026-08-21 23:59:59")
        print(f"[PASS] 区间查询 {len(df)} 条, 列: {list(df.columns)[:5]}...")
        assert len(df) >= 3

        # 3. latest
        lat = db.latest(5)
        print(f"[PASS] latest {len(lat)} 条")
        assert len(lat) >= 3

        # 4. count
        cnt = db.count()
        print(f"[PASS] 总记录 {cnt}")
        assert cnt >= 3

        # 5. 工况过滤
        dfc = db.query(start="2026-08-21 00:00:00", end="2026-08-21 23:59:59",
                       condition="startup")
        print(f"[PASS] 工况过滤 {len(dfc)} 条")
        if len(dfc):
            assert (dfc["operating_condition"] == "startup").all()

        # 6. 分区统计
        st = db.stats()
        print(f"[PASS] 分区: {[p['name'] for p in st['partitions']]}")
        assert any("sensor_ts_2026_08" in p["name"] for p in st["partitions"])
        print("\nTSDB 全部验收通过 ✔")
    finally:
        db.close()


if __name__ == "__main__":
    main()
