"""Eight-fault set metrics and rule reward; no model/client imports.

The reward excludes confidence: synthetic target confidence is not a calibrated
probability. Evidence checks require sensor-specific numerical traceability.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def strict_json(text):
    def bad(value):
        raise ValueError(f"non-finite JSON: {value}")
    return json.loads(text, parse_constant=bad)


def observation(prompt):
    return strict_json(prompt.split("【当前窗口统计】", 1)[1].split("【当前工况】", 1)[0])


def evidence_traceable(evidence, prompt):
    if not evidence:
        return False
    observed = observation(prompt)
    unavailable = prompt.split("【不可用通道】", 1)[1].split("【维修工单】", 1)[0].strip()
    numeric_items = 0
    for item in evidence:
        if item == "缺失通道：" + unavailable and unavailable:
            continue
        sensors = [v for v in observed.values() if v["传感器"] in item]
        if not sensors:
            return False
        # Check each sensor clause separately; a value from another sensor cannot justify it.
        names = "|".join(re.escape(v["传感器"]) for v in observed.values())
        clauses = re.split(f"(?={names})", item)
        checked = False
        for clause in clauses:
            sensor = next((v for v in sensors if clause.startswith(v["传感器"])), None)
            if sensor is None:
                if re.search(r"\d", clause):
                    return False
                continue
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", clause)
            if not numbers:
                return False
            values = [sensor[k] for k in ("均值", "去趋势标准差", "每分钟变化")]
            for number in numbers:
                decimals = len(number.split(".")[1]) if "." in number else 0
                tolerance = max(0.00051, 0.5 * 10 ** -decimals)
                if not any(abs(float(number) - value) <= tolerance for value in values):
                    return False
            checked = True
        if not checked:
            return False
        numeric_items += 1
    return numeric_items > 0


@dataclass
class HardScore:
    total: float = 0.0
    json_ok: bool = False
    correct: bool = False
    set_exact: bool = False
    uncertainty_correct: bool = False
    confidence: float | None = None
    hallu_ok: bool = False
    sample_f1: float = 0.0
    sop_recall: float = 0.0
    predicted_causes: list | None = None
    needs_more_data: bool | None = None


class HardFaultScorer:
    def __init__(self, ontology_path=None):
        self.ontology = strict_json(Path(ontology_path or ROOT / "data/sft_hard_v1/ontology.json").read_text())
        self.names = set(self.ontology["fault_names"].values())
        self.sops = {self.ontology["fault_names"][key]: values for key, values in self.ontology["sops"].items()}

    def parse(self, text):
        value = strict_json(text)
        if not isinstance(value, dict):
            raise ValueError("output must be a JSON object")
        roots = value.get("root_causes")
        if not isinstance(roots, list) or any(not isinstance(r, str) or r not in self.names for r in roots):
            raise ValueError("unknown or invalid causes")
        if len(set(roots)) != len(roots) or type(value.get("needs_more_data")) is not bool:
            raise ValueError("duplicate causes or invalid uncertainty")
        conf = value.get("confidence")
        if type(conf) not in (int, float) or not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("invalid confidence")
        for field in ("evidence", "sop"):
            if not isinstance(value.get(field), list) or any(not isinstance(s, str) or not s.strip() for s in value[field]):
                raise ValueError(f"invalid {field}")
        root = " + ".join(sorted(roots)) if roots else self.ontology[
            "uncertain_label" if value["needs_more_data"] else "normal_label"]
        if value.get("root_cause") != root:
            raise ValueError("root_cause inconsistent with set/uncertainty")
        return value

    def score(self, prompt, completion, gold):
        target = strict_json(gold) if isinstance(gold, str) else gold
        expected = set(target["root_causes"])
        if not expected <= self.names or type(target["needs_more_data"]) is not bool:
            raise ValueError("invalid gold target")
        try:
            value = self.parse(completion)
        except (ValueError, TypeError):
            return HardScore()
        pred = set(value["root_causes"])
        exact = pred == expected
        f1 = 2 * len(pred & expected) / (len(pred) + len(expected)) if pred or expected else 1.0
        uncertain = value["needs_more_data"] == target["needs_more_data"]
        faithful = evidence_traceable(value["evidence"], prompt)
        actions = [s for name in expected for s in self.sops[name]]
        if target["needs_more_data"]:
            actions.append("补充缺失通道或延长观测后再确认其余根因")
        if not actions:
            actions = ["持续监测并核对工况变化"]
        sop = sum(any(action in output for output in value["sop"]) for action in actions) / len(actions)
        total = 0.4 * exact + 0.2 * f1 + 0.2 * uncertain + 0.1 * faithful + 0.05 * sop + 0.05
        return HardScore(total, True, exact and uncertain, exact, uncertain, value["confidence"],
                         faithful, f1, sop, sorted(pred), value["needs_more_data"])


def summarize(results, names):
    n = len(results)
    def avg(key):
        return sum(r["score"][key] for r in results) / n if n else None
    per_class = {}
    for name in sorted(names):
        tp = fp = fn = 0
        for row in results:
            gold = set(row["gold"]["root_causes"])
            pred = set(row["score"]["predicted_causes"] or [])
            tp += name in gold and name in pred
            fp += name not in gold and name in pred
            fn += name in gold and name not in pred
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn,
                           "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    tp = sum(r["gold"]["needs_more_data"] and r["score"]["needs_more_data"] is True for r in results)
    fp = sum(not r["gold"]["needs_more_data"] and r["score"]["needs_more_data"] is True for r in results)
    fn = sum(r["gold"]["needs_more_data"] and r["score"]["needs_more_data"] is not True for r in results)
    normals = [r for r in results if not r["meta"]["injected_faults"]]
    calibrated = [r for r in results if r["score"]["confidence"] is not None]
    return {"n": n, "errors": sum(bool(r.get("error")) for r in results),
            "set_exact": avg("set_exact"), "joint_accuracy": avg("correct"),
            "macro_f1": sum(f1s) / len(f1s) if f1s else None, "per_class": per_class,
            "json_valid": avg("json_ok"), "evidence_traceable": avg("hallu_ok"),
            "sop_recall": avg("sop_recall"), "uncertainty_accuracy": avg("uncertainty_correct"),
            "uncertainty_precision": tp / (tp + fp) if tp + fp else None,
            "uncertainty_recall": tp / (tp + fn) if tp + fn else None,
            "normal_false_positive": sum(bool(r["score"]["predicted_causes"]) for r in normals) / len(normals) if normals else None,
            "normal_count": len(normals), "calibration_count": len(calibrated),
            "brier_joint": sum((r["score"]["confidence"] - r["score"]["correct"]) ** 2 for r in calibrated) / len(calibrated) if calibrated else None}
