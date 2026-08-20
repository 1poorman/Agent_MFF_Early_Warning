"""分级预警推送器。

推送规则（题目 + 技术细节）：
- 红色（一级重大）：声光报警+短信+电话 → 厂长/安全主管/运维班长
- 橙色（二级）：移动端+短信 → 班长/维修工
- 黄色（三级）：移动端 → 值班运维

当前为 mock 通道（记录推送日志），预留真实短信/电话/移动端接口。
"""

from dataclasses import dataclass, field
from typing import Dict, List

# 级别 -> (渠道列表, 接收人列表)
PUSH_MATRIX = {
    "red": {
        "channels": ["sound_light_alarm", "sms", "phone"],
        "receivers": ["厂长", "安全主管", "运维班长"],
    },
    "orange": {
        "channels": ["mobile_app", "sms"],
        "receivers": ["运维班长", "维修工"],
    },
    "yellow": {
        "channels": ["mobile_app"],
        "receivers": ["值班运维"],
    },
}


@dataclass
class PushRecord:
    """单条推送记录。"""
    order_id: str
    level: str
    channel: str
    receiver: str
    content: str
    status: str = "sent"      # sent / failed


class Notifier:
    """分级推送器（mock 通道 + 全链路记录）。"""

    def __init__(self):
        self.records: List[PushRecord] = []

    def push(self, order) -> List[PushRecord]:
        """按工单级别分级推送，返回推送记录。"""
        level = order.level if order.level in PUSH_MATRIX else "yellow"
        cfg = PUSH_MATRIX[level]
        content = f"[{level.upper()}] {order.root_cause} | {order.features}"
        made = []
        for ch in cfg["channels"]:
            for rcv in cfg["receivers"]:
                rec = PushRecord(order.order_id, level, ch, rcv, content)
                self._send(rec)
                made.append(rec)
        self.records.extend(made)
        return made

    def _send(self, rec: PushRecord):
        """实际发送（mock）。真实环境替换为短信网关/电话API/移动端推送。"""
        # TODO: 接入真实通道（短信网关 / 语音呼叫 / APP 推送）
        rec.status = "sent"

    def summary(self) -> Dict:
        return {
            "total": len(self.records),
            "by_level": {lv: sum(1 for r in self.records if r.level == lv) for lv in PUSH_MATRIX},
            "by_channel": {},
        }
