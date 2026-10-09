"""Analysis regressions use small in-memory evaluations, without model artifacts."""
import copy
from dataclasses import asdict
import unittest

from tools.analyze_hard_comparison import (
    GROUPS, audit_summary, cluster_intervals, error_breakdown, require_equal,
)
from tools.hard_fault_metrics import HardScore, summarize


def row(episode, window, *, exact=True, uncertain=True):
    score = HardScore(json_ok=True, correct=exact and uncertain, set_exact=exact,
                      uncertainty_correct=uncertain, predicted_causes=['fault'] if exact else [],
                      needs_more_data=not uncertain, confidence=.7)
    return {'index': window, 'score': asdict(score),
            'gold': {'root_causes': ['fault'], 'needs_more_data': False},
            'meta': {'episode_id': episode, 'window_end': window, 'injected_faults': [],
                     'hidden_channels': [], 'shifted_envelope': False},
            'finish_reason': 'stop', 'error': None}


class HardAnalysisTests(unittest.TestCase):
    def test_paired_identical_methods_have_exact_zero_delta(self):
        observations = [row('a', 1), row('a', 2, exact=False), row('b', 1), row('b', 2)]
        result = cluster_intervals({'lora': [observations] * 3, 'full': [observations] * 3}, resamples=500)
        self.assertEqual(result['episodes'], 2)
        self.assertEqual(result['windows'], 4)
        self.assertEqual(result['methods']['full']['set_exact']['estimate'], .75)
        self.assertEqual(result['paired_vs_lora']['full']['set_exact'], {'estimate': 0., 'ci95': [0., 0.]})

    def test_repeated_inference_does_not_narrow_episode_interval(self):
        observations = [row('a', 1), row('a', 2), row('b', 1, exact=False), row('b', 2, exact=False)]
        single = cluster_intervals({'lora': [observations]}, resamples=500)
        repeated = cluster_intervals({'lora': [observations] * 3}, resamples=500)
        self.assertEqual(single['methods'], repeated['methods'])
        self.assertEqual(single['methods']['lora']['set_exact']['ci95'], [0., 1.])

    def test_unequal_episode_sizes_preserve_window_weighting(self):
        observations = [row('a', 1), row('b', 1, exact=False), row('b', 2, exact=False)]
        result = cluster_intervals({'lora': [observations]}, resamples=500)
        self.assertAlmostEqual(result['methods']['lora']['set_exact']['estimate'], 1 / 3)

    def test_unpaired_or_duplicate_windows_are_rejected(self):
        observations = [row('a', 1), row('b', 1)]
        for evaluations in ({'lora': [observations], 'full': [observations[::-1]]},
                            {'lora': [observations], 'full': [observations] * 2},
                            {'lora': [observations + observations]}, {'lora': []}):
            with self.subTest(evaluations=evaluations), self.assertRaises(ValueError):
                cluster_intervals(evaluations, resamples=10)
        with self.assertRaises(ValueError):
            cluster_intervals({'lora': [observations]}, resamples=0)

    def test_interval_sign_for_consistent_improvement(self):
        baseline = [row('a', 1, exact=False), row('b', 1, exact=False)]
        candidate = [row('a', 1), row('b', 1)]
        result = cluster_intervals({'lora': [baseline], 'full': [candidate]}, resamples=100)
        self.assertEqual(result['paired_vs_lora']['full']['set_exact'], {'estimate': 1., 'ci95': [1., 1.]})

    def test_summary_tampering_is_rejected_including_subgroups(self):
        observations = [row('a', 1), row('b', 1, exact=False)]
        saved = summarize(observations, {'fault'})
        saved['groups'] = {name: summarize([r for r in observations if select(r)], {'fault'})
                           for name, select in GROUPS.items()}
        saved['truncated'] = 0
        audit_summary(observations, {'fault'}, saved, 'fixture')
        for path in [('set_exact',), ('per_class', 'fault', 'tp'), ('groups', 'unshifted', 'n'),
                     ('truncated',)]:
            changed = copy.deepcopy(saved)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] += 1
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'summary mismatch'):
                audit_summary(observations, {'fault'}, changed, 'fixture')
        missing = copy.deepcopy(saved)
        self.assertIsNone(missing.pop('uncertainty_precision'))
        with self.assertRaisesRegex(ValueError, 'missing fields'):
            audit_summary(observations, {'fault'}, missing, 'fixture')

    def test_summary_comparison_rejects_missing_nulls_and_nonfinite(self):
        for actual, expected in (({'n': 2}, {'n': 2, 'f1': None}), (float('nan'), .5),
                                 (float('inf'), .5), (True, 1.), (False, 0)):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                require_equal(actual, expected, 'fixture')

    def test_error_categories_partition_all_predictions(self):
        observations = [row('a', 1), row('a', 2, exact=False), row('b', 1, uncertain=False),
                        row('b', 2, exact=False, uncertain=False), row('c', 1)]
        observations[-1]['score'] = asdict(HardScore())
        errors = error_breakdown([observations])
        self.assertEqual(errors['categories'], {'correct': 1, 'set_only': 1, 'uncertainty_only': 1,
                                               'both': 1, 'invalid': 1})
        self.assertEqual(errors['predictions'], 5)
        self.assertEqual(errors['missed_causes'], {'fault': 2})
        self.assertEqual(errors['uncertainty'], {'tp': 0, 'fp': 2, 'fn': 0, 'tn': 2, 'invalid': 1})
        self.assertEqual(len(errors['failures']), 4)


if __name__ == '__main__':
    unittest.main()
