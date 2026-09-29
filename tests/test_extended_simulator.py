import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from simulator import DataSimulator, FaultSpec, SimConfig
from simulator.physics import PlantModel
from simulator.faults import FAULT_REGISTRY
from tools.build_hard_sft_data import episode, audit


class ExtendedSimulatorTests(unittest.TestCase):
    def test_legacy_outputs_are_identical(self):
        expected={
            'none':'5062145fb58c76b763c962003936e359a2c6a4916a127411280ac611d31aa2ea',
            'filter_clog':'974ddc9b0e202f993e22b8d1db7dbab349f8922a0527ee5c35f7f3a2367cf33f',
            'pump_cavitation':'39eb5807ceacb6422cb3e79ef37528c00068c85b063e8dd0a7b3aca66075f3a9',
            'pipe_leak':'de031369a64cc90fc9dc3dc3a5638f64de3d2153ff40da5a4b39c5c869068463',
            'scale_buildup':'12db97314672f1d0814aa6551ec930c9155c3355e38117f62a839baa502b458d',
        }
        for name,digest in expected.items():
            faults=[] if name=='none' else [FaultSpec(name,300,300,0.6)]
            df=DataSimulator(SimConfig(seed=17),faults=faults).run(1200)
            self.assertEqual(hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),digest,name)

    def test_new_fault_signatures_and_heat_balance(self):
        cfg=SimConfig(seed=19)
        baseline=DataSimulator(cfg).run(1800)
        channels={'heat_exchanger_degradation':'inlet_temp','water_quality':'conductivity',
                  'cabinet_fan_failure':'cabinet_temp','makeup_valve_stuck_open':'tank_level'}
        for name,channel in channels.items():
            with self.subTest(fault=name):
                df=DataSimulator(cfg,faults=[FaultSpec(name,300,300,0.8)]).run(1800)
                self.assertGreater(df[channel].iloc[-120:].mean()-baseline[channel].iloc[-120:].mean(),5)
                self.assertTrue(df.iloc[:300].equals(baseline.iloc[:300]))
                heat=cfg.coil_loss_frac*df.electric_power+cfg.furnace_water_coef*(df.furnace_temp-df.inlet_temp)
                predicted=heat/(cfg.water_cp*df.flow_rate)
                self.assertLess(float(np.median(abs(predicted-(df.outlet_temp-df.inlet_temp)))),0.5)
                self.assertFalse(df.isna().any().any())
        self.assertEqual(len(FAULT_REGISTRY),8)

    def test_stuck_valve_mass_balance_and_overflow(self):
        cfg=SimConfig(tank_level_init=2.99,tank_overflow_level=3.0)
        model=PlantModel(cfg)
        initial=model.state.tank_level
        model.step_tank({'uncommanded_refill_lps':1.5},120)
        stored=(model.state.tank_level-initial)*cfg.tank_area*1000
        spilled=model.state.overflow_lps*120
        self.assertAlmostEqual(stored+spilled,(1.5-0.02)*120)

    def test_dataset_observability_and_isolation(self):
        datasets={split:episode(split,0,['water_quality'],0.8,300,2)
                  for split in ['train','val','eval']}
        audit(datasets)
        for rows in datasets.values():
            self.assertIn('水质恶化',rows[0]['meta']['gold_causes'])
            self.assertEqual(rows[1]['meta']['gold_causes'],[])
            self.assertTrue(rows[1]['meta']['needs_more_data'])
            self.assertNotIn('counterfactual_support',rows[1]['messages'][1]['content'])
        datasets['val']=datasets['train']
        with self.assertRaises(ValueError): audit(datasets)


if __name__=='__main__':
    unittest.main()
