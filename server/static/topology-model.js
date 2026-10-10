/* State comes only from monitoring frames and diagnosis, never the scenario selector. */
(function (root) {
  'use strict';
  const NODES = {
    furnaces: {name:'炉体 / 感应线圈', keys:['furnace_temp','outlet_temp','inlet_temp'], note:'炉一、炉二共用当前测点；异常只能定位到炉体区域，不能区分具体炉次。'},
    power: {name:'中频电源 / 电气柜', keys:['electric_power','electric_current','cabinet_temp','cabinet_humidity'], note:'电源与柜体环境测点用于观察负载、散热和凝露风险。'},
    tank: {name:'循环水箱', keys:['tank_level','conductivity'], note:'水位数值来自采集数据；箱体水面为示意，不表示实际容积比例。'},
    pump: {name:'主水泵', keys:['pressure','flow_rate'], note:'转动按共用流量示意，未接入泵转速反馈；压力震荡仅是气蚀排查线索。'},
    filter: {name:'循环过滤器', keys:['flow_rate','pressure'], note:'按项目监测对象补充的过滤器示意位置；尚无独立过滤器压差测点。'},
    tower: {name:'闭式冷却塔 / 换热器', keys:['inlet_temp','outlet_temp'], note:'展示盘管换热与喷淋散热路径；风机、喷淋泵无独立状态反馈。'},
    makeup: {name:'补水阀 / 液位控制', keys:['tank_level','conductivity'], note:'可观察液位与水质；未采集阀门开度和开关反馈。'},
    pipework: {name:'循环管路 / 供水分支', keys:['pressure','flow_rate','flow_velocity','cabinet_humidity'], note:'共用测点不能确定具体漏点或故障支路；高亮代表排查区域。'}
  };
  const SENSORS = {
    inlet_temp:['进水','℃'],outlet_temp:['出水','℃'],pressure:['压力','kPa'],flow_rate:['流量','L/s'],
    flow_velocity:['流速','m/s'],tank_level:['液位','cm'],conductivity:['电导率','µS/cm'],
    cabinet_temp:['柜温','℃'],cabinet_humidity:['湿度','%RH'],furnace_temp:['炉温','℃'],
    electric_power:['功率','kW'],electric_current:['电流','A']
  };
  const RULES = {
    OUTLET_TEMP_HIGH:['furnaces','回水温度偏高'],DELTA_T_HIGH:['furnaces','进出水温差偏大'],
    COMBO_HEAT_BUILDUP:['furnaces','低流量与高温：积热风险'],INLET_TEMP_HIGH:['tower','供水温度偏高，检查换热'],
    PRESSURE_LOW:['pipework','供水压力偏低，故障待定位'],PRESSURE_HIGH:['pipework','供水压力偏高'],
    FLOW_LOW:['pipework','循环流量不足，故障待定位'],FLOW_HIGH:['pipework','循环流量偏高'],
    PRESSURE_OSC:['pump','压力震荡，疑似水泵气蚀'],CONDUCTIVITY_HIGH:['tank','水质电导率超限'],
    DEW_MARGIN_LOW:['power','电气柜凝露风险'],HUMIDITY_HIGH:['power','柜体环境湿度偏高'],
    COMBO_LEAK_SUSPECT:['pipework','疑似泄漏，具体漏点待核查'],
    PQ_OFFSET:['pipework','管网特性偏移'],HEAT_RATE_DRIFT:['furnaces','热交换表现异常']
  };
  const CAUSES = [
    ['filter',['filter_clog','过滤器堵塞','滤网堵塞']],['pump',['pump_cavitation','水泵气蚀','水泵汽蚀']],
    ['pipework',['pipe_leak','管道泄漏','管路泄漏']],['furnaces',['scale_buildup','线圈结垢']],
    ['tower',['heat_exchanger_degradation','换热器散热衰退','换热器散热异常','换热衰退']],
    ['tank',['water_quality','水质恶化']],['power',['cabinet_fan_failure','柜风机失效','电气柜风机','凝露']],
    ['makeup',['makeup_valve_stuck_open','补水阀卡开']]
  ];
  function number(value) {
    if(value === null || value === undefined || typeof value === 'boolean' || value === '') return null;
    const n=Number(value); return Number.isFinite(n) ? n : null;
  }
  class Model {
    constructor(){ this.threshold=.6; this.reset(); }
    reset(status='idle') {
      this.status=status;this.metrics={};this.timestamp=null;this.received=null;
      this.alerts=[];this.anomaly=null;this.forecastRisk=false;this.dewRisk=false;
      this.diagnosis=null;this.diagnosing=false;this.source=null;
    }
    start(source='simulator'){this.reset('starting');this.source=source;}
    finish(status='stopped'){this.status=status;}
    diagnose(d) {
      if(d==='diagnosing'){this.diagnosing=true;return;}
      if(!d || typeof d!=='object')return;
      this.diagnosis=d;this.diagnosing=false;
    }
    ingest(frame,now) {
      if(!frame || !['starting','running'].includes(this.status))return;
      const metrics=frame.metrics;
      if(metrics && Object.keys(SENSORS).some(k=>Object.prototype.hasOwnProperty.call(metrics,k))){
        this.metrics=Object.fromEntries(Object.keys(SENSORS).map(k=>[k,number(metrics[k])]));
        this.timestamp=frame.timestamp || null;this.received=now;this.status='running';
        // Each sample replaces current rules. Historic diagnoses remain separately labelled.
        this.alerts=Array.isArray(frame.l1) ? frame.l1 : [];
        this.anomaly=number(frame.l2?.anomaly_score);
        this.forecastRisk=number(frame.l2?.exceed_eta)!==null;
        this.dewRisk=Boolean(frame.dew?.at_risk || number(frame.dew?.eta_s)!==null);
      }
      if(frame.l3)this.diagnose(frame.l3);
    }
    snapshot(now) {
      const age=this.received===null ? null : Math.max(0,now-this.received);
      const fresh=this.status==='running' && age!==null && age<=30000;
      const status=this.status==='running' && !fresh ? 'stale' : this.status;
      const observations=[];
      for(const alert of this.alerts){
        const mapped=RULES[alert.rule_id];
        observations.push({node:mapped?.[0] || null, text:mapped?.[1] || String(alert.message || alert.rule_id), level:'warning', kind:'当前预警'});
      }
      if(this.dewRisk && !observations.some(x=>x.node==='power'))observations.push({node:'power',text:'凝露风险，检查柜体环境',level:'warning',kind:'当前预警'});
      if(this.forecastRisk)observations.push({node:'furnaces',text:'预测回水温度将越限',level:'warning',kind:'趋势预警'});
      if(this.anomaly>this.threshold && !observations.length)observations.push({node:null,text:'多参数异常，尚未定位设备',level:'warning',kind:'L2 异常'});
      const findings=[];
      if(this.diagnosis){
        const d=this.diagnosis;
        const causes=Array.isArray(d.root_causes) ? d.root_causes : [d.root_cause];
        const normal=causes.length===0 || causes.every(c=>/^(正常|正常工况|无故障|normal)$/i.test(String(c)));
        if(!normal){
          const matched=new Set();
          for(const [node,aliases] of CAUSES){
            if(causes.some(c=>aliases.some(a=>String(c).includes(a))))matched.add(node);
          }
          const text=causes.filter(Boolean).join('、') || '尚未明确根因';
          const level=d.manual_required || d.needs_more_data ? 'warning' : d.level==='red' ? 'critical' : 'fault';
          if(matched.size)for(const node of matched)findings.push({node,text,level,kind:'本轮诊断'});
          else findings.push({node:null,text,level:'warning',kind:'本轮诊断'});
        }
      }
      const issues=[...findings,...observations];
      const nodes=Object.fromEntries(Object.keys(NODES).map(id=>[id,{level:'idle',issues:[]}]));
      const ranks={idle:0,warning:1,fault:2,critical:3};
      for(const issue of issues){
        if(!issue.node)continue;
        const node=nodes[issue.node];node.issues.push(issue);
        if(ranks[issue.level]>ranks[node.level])node.level=issue.level;
      }
      const flow=this.metrics.flow_rate;
      return {status,fresh,age,timestamp:this.timestamp,metrics:{...this.metrics},source:this.source,
        flowing:fresh && flow!==null && flow!==undefined && flow>0,
        flowSeconds:flow>0 ? Math.max(.6,Math.min(8,2.4*8/flow)) : 0,
        nodes,issues,diagnosing:this.diagnosing,diagnosis:this.diagnosis};
    }
  }
  const api={Model,NODES,SENSORS};
  if(typeof module!=='undefined' && module.exports)module.exports=api;
  else root.MFFTopology=api;
})(typeof window==='undefined' ? globalThis : window);
