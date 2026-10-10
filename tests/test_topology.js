const {test}=require('node:test');
const assert=require('node:assert/strict');
const {Model}=require('../server/static/topology-model.js');
const sample=(metrics={},l1=[],extra={})=>({timestamp:'2026-10-10 10:00:00',metrics:{flow_rate:8,inlet_temp:28,outlet_temp:42,pressure:230,...metrics},l1,l2:{anomaly_score:.2},...extra});
const active=()=>{const m=new Model();m.start();return m;};

test('preview and preload cannot become live observations or start the flow',()=>{
  const m=new Model();m.reset('preview');m.ingest(sample(),0);
  assert.equal(m.snapshot(0).flowing,false);assert.equal(m.snapshot(0).timestamp,null);
  m.start();m.ingest({preload:true,points:1000},0);
  assert.equal(m.snapshot(0).status,'starting');assert.equal(m.snapshot(0).flowing,false);
});
test('fresh measured flow drives animation; stopped, stale, missing and zero flow freeze it',()=>{
  const m=active();m.ingest(sample(),0);
  assert.equal(m.snapshot(0).flowing,true);
  const fast=m.snapshot(0).flowSeconds;
  m.ingest(sample({flow_rate:2}),100);assert.ok(m.snapshot(100).flowSeconds>fast);
  assert.equal(m.snapshot(30101).flowing,false);assert.equal(m.snapshot(30101).status,'stale');
  m.ingest(sample({flow_rate:0}),30200);assert.equal(m.snapshot(30200).flowing,false);
  m.ingest(sample({flow_rate:NaN}),30300);assert.equal(m.snapshot(30300).metrics.flow_rate,null);
  assert.equal(m.snapshot(30300).flowing,false);
  m.ingest(sample(),30400);m.finish();assert.equal(m.snapshot(30400).flowing,false);
  m.ingest(sample({flow_rate:9}),30500);assert.equal(m.snapshot(30500).metrics.flow_rate,8);
});
test('low flow marks only the common circuit and does not guess a blocked filter',()=>{
  const m=active();m.ingest(sample({flow_rate:4},[{rule_id:'FLOW_LOW'}]),0);
  const s=m.snapshot(0);
  assert.equal(s.nodes.pipework.level,'warning');assert.equal(s.nodes.filter.level,'idle');
  assert.equal(s.nodes.pump.level,'idle');
});
test('pressure oscillation and leakage rules expose suspected areas immediately',()=>{
  const m=active();m.ingest(sample({},[{rule_id:'PRESSURE_OSC'},{rule_id:'COMBO_LEAK_SUSPECT'}]),0);
  assert.equal(m.snapshot(0).nodes.pump.level,'warning');
  assert.match(m.snapshot(0).nodes.pipework.issues[0].text,/漏点待核查/);
});
test('diagnosis maps the four demo faults and additional supported findings to components',()=>{
  for(const [root_cause,node] of Object.entries({过滤器堵塞:'filter',水泵气蚀:'pump',管道泄漏:'pipework',线圈结垢:'furnaces',heat_exchanger_degradation:'tower',water_quality:'tank',cabinet_fan_failure:'power',makeup_valve_stuck_open:'makeup'})){
    const m=active();m.ingest(sample(),0);m.diagnose({root_cause,level:'orange',confidence:.9});
    const s=m.snapshot(0);assert.equal(s.nodes[node].level,'fault',root_cause);
    assert.equal(s.nodes[node].issues[0].kind,'本轮诊断');
    assert.equal(s.nodes.furnace1,undefined);assert.equal(s.nodes.furnace2,undefined);
  }
});
test('L2 anomalies without location and unknown diagnoses do not paint arbitrary equipment',()=>{
  const m=active();m.threshold=.8;m.ingest(sample({},[],{l2:{anomaly_score:.7}}),0);
  assert.equal(m.snapshot(0).issues.length,0);
  m.ingest(sample({},[],{l2:{anomaly_score:.9}}),1);
  assert.equal(m.snapshot(1).issues[0].node,null);
  m.diagnose({root_cause:'待现场检查',manual_required:true});
  assert.ok(Object.values(m.snapshot(1).nodes).every(n=>n.level==='idle'));
});
test('metadata-only diagnosis frames preserve measurements and freshness age',()=>{
  const m=active();m.ingest(sample(),0);
  m.ingest({timestamp:'later',metrics:{},l2:{anomaly_score:0},l3:{root_cause:'过滤器堵塞',level:'orange'}},25000);
  const s=m.snapshot(30001);assert.equal(s.status,'stale');assert.equal(s.metrics.flow_rate,8);
  assert.equal(s.timestamp,'2026-10-10 10:00:00');assert.equal(s.nodes.filter.level,'fault');
});
test('live rules clear on recovery; the separately labelled diagnosis persists until reset',()=>{
  const m=active();m.ingest(sample({},[{rule_id:'FLOW_LOW'}]),0);
  m.diagnose({root_cause:'过滤器堵塞',level:'orange'});m.ingest(sample(),1);
  assert.equal(m.snapshot(1).nodes.pipework.level,'idle');assert.equal(m.snapshot(1).nodes.filter.level,'fault');
  m.start('upload');assert.equal(m.snapshot(2).issues.length,0);assert.equal(m.snapshot(2).timestamp,null);
});
test('manual verification and combined diagnoses retain their uncertainty and all regions',()=>{
  const m=active();m.ingest(sample(),0);
  m.diagnose({root_causes:['pipe_leak','pump_cavitation'],level:'red',needs_more_data:true});
  assert.equal(m.snapshot(0).nodes.pipework.level,'warning');assert.equal(m.snapshot(0).nodes.pump.level,'warning');
  m.diagnose({root_cause:'正常',level:'green'});assert.equal(m.snapshot(0).issues.length,0);
});
test('missing sensor values replace previous values and never display old readings as current',()=>{
  const m=active();m.ingest(sample(),0);m.ingest({metrics:{flow_rate:3},l1:[]},1);
  assert.equal(m.snapshot(1).metrics.pressure,null);assert.equal(m.snapshot(1).metrics.flow_rate,3);
});
