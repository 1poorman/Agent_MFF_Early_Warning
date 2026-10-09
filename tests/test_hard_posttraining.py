import copy
from dataclasses import asdict
import json
from pathlib import Path
import unittest

from tools.hard_fault_metrics import HardFaultScorer, evidence_traceable, summarize
from tools.rl_reward import make_reward_func


class HardPosttrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = HardFaultScorer()
        rows = [json.loads(s) for s in Path('data/sft_hard_v1/train.jsonl').read_text().splitlines()]
        cls.row = next(r for r in rows if len(r['meta']['gold_causes']) == 2)
        cls.answer = json.loads(cls.row['messages'][-1]['content'])
        cls.gold = {k: cls.answer[k] for k in ('root_causes', 'needs_more_data')}
        cls.prompt = cls.row['messages'][1]['content']

    def score(self, answer):
        return self.scorer.score(self.prompt, json.dumps(answer, ensure_ascii=False), self.gold)

    def test_set_order_and_confidence_do_not_change_reward(self):
        answer = copy.deepcopy(self.answer)
        answer['root_causes'].reverse()
        answer['confidence'] = 0.01
        self.assertAlmostEqual(self.score(answer).total, 1)
        self.assertTrue(self.score(answer).correct)

    def test_omission_and_uncertainty_are_distinct_errors(self):
        answer = copy.deepcopy(self.answer)
        answer['root_causes'].pop()
        answer['root_cause'] = ' + '.join(answer['root_causes'])
        omitted = self.score(answer)
        self.assertFalse(omitted.set_exact)
        self.assertAlmostEqual(omitted.sample_f1, 2/3)
        answer = copy.deepcopy(self.answer)
        answer['needs_more_data'] = not answer['needs_more_data']
        uncertain = self.score(answer)
        self.assertTrue(uncertain.set_exact)
        self.assertFalse(uncertain.correct)
        self.assertLess(uncertain.total, 1)

    def test_invalid_types_unknowns_and_duplicates_are_zero(self):
        for key, value in [('confidence', float('nan')), ('confidence', True),
                           ('root_causes', ['不存在的故障']), ('needs_more_data', 'false'),
                           ('root_causes', self.answer['root_causes'] * 2), ('evidence', [None])]:
            with self.subTest(key=key, value=value):
                self.assertEqual(self.score({**self.answer, key: value}).total, 0)

    def test_evidence_must_match_the_named_sensor(self):
        observed = {'inlet_temp': {'传感器': '进水温度', '均值': 30.123, '去趋势标准差': .1, '每分钟变化': .02},
                    'flow_rate': {'传感器': '流量', '均值': 8.567, '去趋势标准差': .03, '每分钟变化': .01}}
        prompt = '【当前窗口统计】' + json.dumps(observed, ensure_ascii=False) + '【当前工况】holding\n【不可用通道】\n【维修工单】'
        self.assertTrue(evidence_traceable(['进水温度为 30.12℃，流量为 8.57 L/s'], prompt))
        self.assertFalse(evidence_traceable(['进水温度为 8.567℃'], prompt))
        self.assertFalse(evidence_traceable(['进水温度为 99999℃'], prompt))
        self.assertFalse(evidence_traceable([], prompt))

    def test_grpo_callback_and_summary_use_same_score(self):
        reward = make_reward_func({}, self.scorer)
        text = json.dumps(self.answer, ensure_ascii=False)
        values = reward(['rendered prompt'], [text], gold=[json.dumps(self.gold)], user_text=[self.prompt])
        self.assertAlmostEqual(values[0], 1)
        results = [{'gold': self.gold, 'score': asdict(self.score(self.answer)), 'meta': self.row['meta'], 'error': None}]
        summary = summarize(results, self.scorer.names)
        self.assertEqual(summary['set_exact'], 1)
        self.assertEqual(summary['macro_f1'], 1)
        self.assertEqual(summary['joint_accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
