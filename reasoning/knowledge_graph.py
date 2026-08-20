"""五域知识图谱：设备-部件-传感器-故障-处置（源自 docs/故障处理.md）。

用有向图存储因果关系，支持：
- 正向推理：故障现象 -> 可能根因（沿因果链回溯）
- 交叉验证：校验 LLM 推理路径的每条因果关系是否在图谱中存在（防幻觉第二层）
- 证据链输出：给出可追溯的推理路径
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# 五域节点类型
DEVICE, COMPONENT, SENSOR, FAULT, ACTION = "device", "component", "sensor", "fault", "action"


@dataclass
class Node:
    id: str
    type: str
    name: str


@dataclass
class Edge:
    src: str
    dst: str
    relation: str  # causes / indicates / monitored_by / handled_by / part_of


class KnowledgeGraph:
    """中频炉水冷系统知识图谱（内存有向图）。"""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._build()

    # ---------------- 构图 ----------------

    def add(self, nid: str, ntype: str, name: str):
        self.nodes[nid] = Node(nid, ntype, name)

    def link(self, src: str, dst: str, relation: str):
        assert src in self.nodes and dst in self.nodes, f"边端点未定义: {src}->{dst}"
        self.edges.append(Edge(src, dst, relation))

    def _build(self):
        A = self.add
        L = self.link

        # 设备与部件
        A("mff", DEVICE, "中频炉")
        A("cooling_sys", DEVICE, "水冷系统")
        A("coil", COMPONENT, "感应线圈")
        A("pump", COMPONENT, "循环水泵")
        A("filter", COMPONENT, "Y型过滤器")
        A("valve3", COMPONENT, "3号阀")
        A("tank", COMPONENT, "冷却水箱")
        A("heat_exchanger", COMPONENT, "板式换热器")
        A("cabinet", COMPONENT, "电气柜")
        A("pipe", COMPONENT, "冷却水管路")
        A("resin", COMPONENT, "离子交换树脂")
        for c in ["coil", "pump", "filter", "valve3", "tank", "heat_exchanger", "pipe", "resin"]:
            L(c, "cooling_sys", "part_of")
        L("cabinet", "mff", "part_of")

        # 传感器
        A("s_outlet_temp", SENSOR, "出水温度")
        A("s_inlet_temp", SENSOR, "进水温度")
        A("s_pressure", SENSOR, "压力")
        A("s_flow", SENSOR, "流量")
        A("s_level", SENSOR, "水箱液位")
        A("s_conductivity", SENSOR, "电导率")
        A("s_humidity", SENSOR, "湿度")
        A("s_cabinet_temp", SENSOR, "电气柜温度")

        # 故障
        A("f_filter_clog", FAULT, "过滤器堵塞")
        A("f_pump_cavitation", FAULT, "水泵气蚀")
        A("f_pipe_leak", FAULT, "管道泄漏")
        A("f_scale", FAULT, "线圈结垢")
        A("f_coil_burn", FAULT, "线圈烧穿")
        A("f_condensation", FAULT, "电气柜凝露")
        A("f_water_quality", FAULT, "水质恶化")
        A("f_elec_breakdown", FAULT, "电气击穿")

        # 处置
        A("a_clean_filter", ACTION, "清洗过滤器")
        A("a_switch_pump", ACTION, "切换备用泵")
        A("a_locate_leak", ACTION, "测漏仪检查管路")
        A("a_acid_wash", ACTION, "弱酸循环清洗除垢")
        A("a_replace_water", ACTION, "更换去离子水")
        A("a_emergency_tilt", ACTION, "紧急倾炉")
        A("a_seal_ring", ACTION, "备件DN50密封圈")

        # ---- 因果关系（故障 -> 传感器现象，fault indicates sensor anomaly）----
        # 过滤器堵塞 -> 流量降/压力变化/出水温升
        L("f_filter_clog", "s_flow", "indicates")
        L("f_filter_clog", "s_outlet_temp", "indicates")
        # 水泵气蚀 -> 压力/流量震荡
        L("f_pump_cavitation", "s_pressure", "indicates")
        L("f_pump_cavitation", "s_flow", "indicates")
        # 管道泄漏 -> 压力降/液位降/湿度升
        L("f_pipe_leak", "s_pressure", "indicates")
        L("f_pipe_leak", "s_level", "indicates")
        L("f_pipe_leak", "s_humidity", "indicates")
        # 线圈结垢 -> 出水温升/温差大/电耗异常
        L("f_scale", "s_outlet_temp", "indicates")
        # 凝露 -> 湿度高/柜温低
        L("f_condensation", "s_humidity", "indicates")
        L("f_condensation", "s_cabinet_temp", "indicates")
        # 水质恶化 -> 电导率升
        L("f_water_quality", "s_conductivity", "indicates")
        # 故障传播链
        L("f_filter_clog", "f_coil_burn", "causes")
        L("f_scale", "f_coil_burn", "causes")
        L("f_pipe_leak", "f_condensation", "causes")
        L("f_water_quality", "f_elec_breakdown", "causes")

        # 部件-故障 关联
        L("f_filter_clog", "filter", "located_at")
        L("f_pump_cavitation", "pump", "located_at")
        L("f_pipe_leak", "pipe", "located_at")
        L("f_pipe_leak", "valve3", "located_at")
        L("f_scale", "coil", "located_at")
        L("f_condensation", "cabinet", "located_at")
        L("f_water_quality", "resin", "located_at")

        # 故障-处置
        L("f_filter_clog", "a_clean_filter", "handled_by")
        L("f_filter_clog", "a_switch_pump", "handled_by")
        L("f_pump_cavitation", "a_switch_pump", "handled_by")
        L("f_pipe_leak", "a_locate_leak", "handled_by")
        L("f_pipe_leak", "a_seal_ring", "handled_by")
        L("f_scale", "a_acid_wash", "handled_by")
        L("f_water_quality", "a_replace_water", "handled_by")
        L("f_coil_burn", "a_emergency_tilt", "handled_by")

    # ---------------- 查询 ----------------

    def _neighbors(self, nid: str, relation: Optional[str] = None, reverse: bool = False) -> List[str]:
        out = []
        for e in self.edges:
            if relation and e.relation != relation:
                continue
            if not reverse and e.src == nid:
                out.append(e.dst)
            elif reverse and e.dst == nid:
                out.append(e.src)
        return out

    def faults_for_sensors(self, sensor_names: List[str]) -> List[Tuple[str, int]]:
        """给定异常传感器名集合，返回候选根因及命中数（倒排：哪些故障 indicates 这些传感器）。"""
        name2id = {n.name: n.id for n in self.nodes.values() if n.type == SENSOR}
        score: Dict[str, int] = {}
        for sname in sensor_names:
            sid = name2id.get(sname)
            if not sid:
                continue
            for fid in self._neighbors(sid, "indicates", reverse=True):
                score[fid] = score.get(fid, 0) + 1
        return sorted(score.items(), key=lambda x: -x[1])

    def causal_path_exists(self, fault_name: str, sensor_name: str) -> bool:
        """校验 故障->传感器 的因果关系是否在图谱中（防幻觉交叉验证）。"""
        name2id = {n.name: n.id for n in self.nodes.values()}
        fid, sid = name2id.get(fault_name), name2id.get(sensor_name)
        if not fid or not sid:
            return False
        return sid in self._neighbors(fid, "indicates")

    def actions_for_fault(self, fault_name: str) -> List[str]:
        name2id = {n.name: n.id for n in self.nodes.values()}
        fid = name2id.get(fault_name)
        if not fid:
            return []
        return [self.nodes[a].name for a in self._neighbors(fid, "handled_by")]

    def fault_names(self) -> Set[str]:
        return {n.name for n in self.nodes.values() if n.type == FAULT}

    def sensor_names(self) -> Set[str]:
        return {n.name for n in self.nodes.values() if n.type == SENSOR}
