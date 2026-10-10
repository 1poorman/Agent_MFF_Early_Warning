"""QA regressions with real API/stream handlers, without GPUs or external services."""
from copy import deepcopy
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from config.settings import RuleSettings
from context.maintenance import WorkOrder, default_maintenance_log
from server.monitoring import MonitorState
from server.qa import KnowledgeQAAgent


def service():
    return SimpleNamespace(monitor=MonitorState(), maint=default_maintenance_log(),
                           cfg=SimpleNamespace(rules=RuleSettings()), use_llm=False)


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.qa = KnowledgeQAAgent(self.svc)

    def test_supported_questions_work_without_llm(self):
        cases = {'监测传感器有哪些？': 'sensors', '当前是什么工况？': 'current',
                 '压力是多少？': 'current', '目前的报警阈值是多少？': 'thresholds',
                 '过滤器堵塞怎么处理？': 'procedure', '查找3号阀的历史维修记录': 'maintenance'}
        for question, topic in cases.items():
            with self.subTest(question=question):
                result = self.qa.ask(question)
                self.assertEqual(result['topic'], topic)
                self.assertTrue(result['sources'])
        self.assertEqual(len(self.qa.ask('监测传感器有哪些？')['cards'][0]['rows']), 12)

    def test_current_observation_is_sanitized_and_becomes_stale(self):
        self.assertFalse(self.qa.ask('当前工况')['cards'])
        sid = self.svc.monitor.start('upload', 'sample.csv')
        row = {'timestamp': '2026-08-20 12:00:00', 'operating_condition': 'holding',
               'pressure': 201.25, 'outlet_temp': float('nan'), 'flow_rate': float('inf')}
        with patch('server.monitoring.time.monotonic', return_value=100):
            self.svc.monitor.record(row, sid)
            result = self.qa.ask('当前工况')
        self.assertTrue(result['snapshot']['fresh'])
        self.assertEqual(result['snapshot']['condition_label'], '保温')
        self.assertIsNone(result['snapshot']['metrics']['outlet_temp'])
        self.assertIsNone(result['snapshot']['metrics']['flow_rate'])
        with patch('server.monitoring.time.monotonic', return_value=131):
            self.assertFalse(self.qa.ask('当前工况')['snapshot']['fresh'])
        self.svc.monitor.finish(sid)
        self.assertIn('最近一次观测', self.qa.ask('当前工况')['answer'])

    def test_old_session_cannot_overwrite_new_session(self):
        old = self.svc.monitor.start()
        with self.assertRaises(ValueError):
            self.svc.monitor.start()
        self.svc.monitor.finish(old)
        new = self.svc.monitor.start()
        self.svc.monitor.record({'timestamp': 'old'}, old)
        self.svc.monitor.finish(old, 'error')
        snapshot = self.svc.monitor.snapshot()
        self.assertEqual(snapshot['session_id'], new)
        self.assertEqual(snapshot['status'], 'starting')
        self.assertIsNone(snapshot['timestamp'])

    def test_maintenance_filters_and_sample_provenance_are_read_only(self):
        self.svc.maint.orders.append(WorkOrder('WO-20261001-10', '2026-10-01', '3号阀', '检查'))
        before = deepcopy(self.svc.maint.orders)
        result = self.qa.ask('查找3号阀的历史维修记录')
        self.assertEqual(result['total'], 2)
        self.assertEqual([c['source'] for c in result['cards']], ['运行台账', '内置示例'])
        result = self.qa.ask('维修记录', keyword='3号阀', since='2026-06-20', until='2026-06-20')
        self.assertEqual(result['cards'][0]['order_id'], 'WO-20260620-04')
        self.assertEqual(self.svc.maint.orders, before)
        self.assertEqual(self.qa.ask('查询2026年7月的维修记录')['total'], 3)
        self.assertEqual(self.qa.ask('查找不存在部件维修记录')['total'], 0)

    def test_followups_keep_subject_but_refresh_dates_and_facts(self):
        result = self.qa.ask('它怎么处理？', ['过滤器堵塞怎么处理？'])
        self.assertEqual(result['cards'][0]['title'], '过滤器堵塞')
        result = self.qa.ask('它的维修记录呢？', ['过滤器堵塞怎么处理？'])
        self.assertEqual(result['filters']['keyword'], '过滤器')
        self.assertEqual(result['total'], 1)
        sid = self.svc.monitor.start('upload')
        self.svc.monitor.record({'timestamp': '2026-07-20 12:00:00'}, sid)
        result = self.qa.ask('那上个月呢？', ['查询2026年7月3号阀的维修记录'])
        self.assertEqual(result['filters']['since'], '2026-06-01')
        self.assertEqual(result['filters']['keyword'], '3号阀')
        self.assertEqual(result['total'], 1)
        result = self.qa.ask('它的最新维修记录呢？', ['查找3号阀维修记录', '那上个月呢？'])
        self.assertEqual(result['filters']['keyword'], '3号阀')
        self.assertEqual(result['total'], 1)
        self.svc.monitor.record({'timestamp': '2026-08-01 12:00:00'}, sid)
        self.assertEqual(self.qa.ask('当前工况', ['当前工况'])['snapshot']['timestamp'], '2026-08-01 12:00:00')

    def test_semantic_router_is_only_allowed_to_select_grounded_tools(self):
        qa = KnowledgeQAAgent(self.svc, router=lambda *_: {'topic': 'procedure', 'subject': '凭空部件'})
        self.assertEqual(qa.ask('你好')['topic'], 'unknown')
        qa.router = lambda *_: {'topic': 'sensors', 'subject': '', 'answer': '虚构答案'}
        result = qa.ask('看看设备有什么探头')
        self.assertEqual(result['engine'], 'semantic-routing')
        self.assertEqual(len(result['cards'][0]['rows']), 12)
        self.assertNotIn('虚构', result['answer'])
        def fail(*_):
            raise TimeoutError('offline')
        qa.router = fail
        self.assertEqual(qa.ask('你好')['topic'], 'unknown')

    def test_thresholds_follow_service_configuration(self):
        self.svc.cfg.rules.outlet_temp_high = 62
        self.assertEqual(self.qa.ask('报警阈值')['cards'][0]['rows'][0][1], '> 62 ℃')


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from server import api
        # MCP's session manager has one lifespan per application instance.
        cls.stack = ExitStack()
        cls.addClassCleanup(cls.stack.close)
        cls.client = cls.stack.enter_context(TestClient(api.app))

    def setUp(self):
        from server import api
        from server.service import AgentService
        self.svc = service()
        self.svc.pipeline = SimpleNamespace(forecast_engine=None)
        self.svc.uploaded_data = [{'timestamp': '2026-08-20 12:00:00',
                                   'operating_condition': 'melting', 'pressure': 205.5}]
        self.svc.uploaded_meta = {'filename': 'test.csv'}
        self.svc.reset_stream = lambda: AgentService.reset_stream(self.svc)
        self.svc.stream_step = lambda row: {'timestamp': row['timestamp'], 'metrics': {'pressure': row['pressure']}}
        self.svc.reset_stream()
        self.patch = patch.object(api, '_service', self.svc)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_api_input_validation_and_static_entry(self):
        for payload in ({'question': ''}, {'question': ' '}, {'question': 'x' * 1001},
                        {'question': '维修', 'since': 'bad'},
                        {'question': '维修', 'since': '2026-07-01', 'until': '2026-06-01'}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post('/api/v1/agents/knowledge-qa/ask', json=payload).status_code, 422)
        result = self.client.post('/api/v1/agents/knowledge-qa/ask', json={'question': '查找3号阀的历史维修记录'})
        self.assertEqual(result.json()['data']['total'], 1)
        self.assertIn('qa-panel', self.client.get('/').text)
        self.assertEqual(self.client.get('/static/qa.js').status_code, 200)
        self.assertEqual(self.client.get('/static/qa.css').status_code, 200)

    def test_preview_and_preload_do_not_become_current_observations(self):
        from server.service import AgentService
        AgentService.preload_row(self.svc, self.svc.uploaded_data[0])
        self.assertIsNone(self.client.get('/api/v1/monitoring/status').json()['data']['timestamp'])

    def test_uploaded_replay_updates_snapshot_and_finishes(self):
        with self.client.websocket_connect('/ws/stream') as ws:
            ws.send_json({'data_source': 'upload', 'speed': 100})
            frames = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame.get('done'):
                    break
        snapshot = self.client.get('/api/v1/monitoring/status').json()['data']
        self.assertEqual(snapshot['status'], 'completed')
        self.assertEqual(snapshot['filename'], 'test.csv')
        self.assertEqual(snapshot['metrics']['pressure'], 205.5)
        self.assertEqual(snapshot['condition'], 'melting')
        self.assertFalse(snapshot['fresh'])
        self.assertTrue(any(f.get('metrics') for f in frames))
        self.assertEqual(self.client.post('/api/v1/stream/reset').status_code, 200)
        self.assertEqual(self.svc.monitor.snapshot()['status'], 'idle')

    def test_active_session_cannot_be_reset_or_replaced_by_another_client(self):
        sid = self.svc.monitor.start()
        self.assertEqual(self.client.post('/api/v1/stream/reset').status_code, 409)
        with self.client.websocket_connect('/ws/stream') as ws:
            ws.send_json({'data_source': 'upload'})
            self.assertIn('已有监测', ws.receive_json()['error'])
        self.assertEqual(self.svc.monitor.snapshot()['session_id'], sid)
        self.assertEqual(self.svc.monitor.snapshot()['status'], 'starting')

    def test_missing_upload_finishes_with_error_and_no_readings(self):
        self.svc.uploaded_data = []
        with self.client.websocket_connect('/ws/stream') as ws:
            ws.send_json({'data_source': 'upload'})
            self.assertIn('未找到已上传数据', ws.receive_json()['error'])
            self.assertTrue(ws.receive_json()['done'])
        snapshot = self.svc.monitor.snapshot()
        self.assertEqual(snapshot['status'], 'error')
        self.assertIsNone(snapshot['timestamp'])


if __name__ == '__main__':
    unittest.main()
