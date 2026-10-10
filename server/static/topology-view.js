(() => {
  'use strict';
  const {Model,NODES,SENSORS}=window.MFFTopology;
  const model=new Model(), byId=id=>document.getElementById(id);
  const panel=byId('topology-view'), drawing=byId('topology-drawing'), dialog=byId('topology-dialog');
  const anchor=document.createComment('topology home'); panel.before(anchor);
  let selected='pipework', pinned=false, queued=false, lastIssueKey='';
  const now=()=>performance.now();
  const text=(id,value)=>{const el=byId(id);if(el && el.textContent!==value)el.textContent=value;};
  const reading=(key,value)=>value===null || value===undefined ? '-- '+SENSORS[key][1] : value.toFixed(1)+' '+SENSORS[key][1];
  const statusNames={idle:'等待监测',starting:'正在准备回放',running:'监测中',stopped:'已停止 · 最近观测',completed:'回放完成 · 最近观测',stale:'数据超时 · 已冻结',error:'连接异常 · 已冻结',preview:'文件预览 · 尚未监测'};
  function schedule(){if(!queued){queued=true;requestAnimationFrame(()=>{queued=false;render();});}}
  function choose(id){if(NODES[id]){selected=id;pinned=true;schedule();}}
  function render(){
    const s=model.snapshot(now());
    panel.classList.toggle('is-flowing',s.flowing);
    panel.classList.toggle('is-frozen',!s.fresh);
    panel.style.setProperty('--flow-time',s.flowSeconds+'s');
    panel.dataset.status=s.status;
    text('topology-status',statusNames[s.status] || s.status);
    text('topology-time',s.timestamp ? '数据时间 '+s.timestamp : '暂无本轮观测');
    let flow='水流动画按共用流量示意';
    if(s.status==='running' && !s.flowing)flow=s.metrics.flow_rate===null ? '流量缺失 · 水流动画暂停' : '流量为零或反向 · 水流动画暂停';
    if(s.status==='stale')flow='超过 30 秒未更新，保留最后一次观测';
    text('topology-flow-note',flow);
    if(!pinned)selected=s.issues.find(x=>x.node)?.node || 'pipework';
    for(const el of drawing.querySelectorAll('[data-value]')){
      const key=el.dataset.value,value=reading(key,s.metrics[key]);
      if(el.textContent!==value)el.textContent=value;
    }
    for(const el of drawing.querySelectorAll('[data-node]')){
      const node=s.nodes[el.dataset.node];
      el.dataset.level=node.level;el.classList.toggle('is-selected',el.dataset.node===selected);
      el.setAttribute('aria-pressed',String(el.dataset.node===selected));
    }
    for(const el of drawing.querySelectorAll('[data-state]')){
      const n=s.nodes[el.dataset.state];
      const label=n.level==='idle' ? (s.timestamp ? '未指向故障' : '未观测') : n.issues.some(x=>x.kind==='本轮诊断') ? '诊断指向' : '预警待核查';
      if(el.textContent!==label)el.textContent=label;
    }
    const issueKey=JSON.stringify([s.issues,s.diagnosing,s.fresh,Boolean(s.timestamp)]);
    if(issueKey!==lastIssueKey){
      lastIssueKey=issueKey;
      const alerts=byId('topology-alerts');alerts.replaceChildren();
      if(!s.issues.length){
        alerts.textContent=s.diagnosing ? '正在进行根因诊断…' : s.timestamp ? '最近观测未收到定位预警；设备详情可点击查看。' : '开始监测后，预警与诊断会同步标出排查区域。';
      }
      const seen=new Set();
      for(const issue of s.issues){
        const label=issue.kind+' · '+issue.text;
        if(seen.has(label))continue;seen.add(label);
        const item=document.createElement(issue.node ? 'button' : 'span');
        item.className='topology-issue '+issue.level;
        item.textContent=(!s.fresh && s.timestamp && issue.kind!=='本轮诊断' ? '最近观测 · ' : '')+label;
        if(issue.node){item.type='button';item.addEventListener('click',()=>choose(issue.node));}
        alerts.append(item);
      }
    }
    const meta=NODES[selected],state=s.nodes[selected];
    text('topology-detail-title',meta.name);
    text('topology-detail-note',meta.note);
    text('topology-detail-evidence',state.issues.length ? state.issues.map(i=>i.kind+'：'+i.text).join('；') : '当前未收到指向此设备的定位信息。');
    const metrics=byId('topology-detail-metrics');
    const metricText=meta.keys.map(k=>SENSORS[k][0]+' '+reading(k,s.metrics[k])).join('  ·  ');
    if(metrics.textContent!==metricText)metrics.textContent=metricText;
  }
  drawing.addEventListener('click',event=>choose(event.target.closest('[data-node]')?.dataset.node));
  drawing.addEventListener('keydown',event=>{
    if(event.key==='Enter' || event.key===' '){event.preventDefault();choose(event.target.closest('[data-node]')?.dataset.node);}
  });
  const tabs=[byId('topology-tab'),byId('curves-tab')];
  function tab(index){
    tabs.forEach((button,i)=>{button.setAttribute('aria-selected',String(i===index));button.tabIndex=i===index ? 0 : -1;});
    panel.hidden=index!==0;byId('curves-view').hidden=index!==1;
    window.dispatchEvent(new Event('resize'));
  }
  tabs.forEach((button,i)=>{
    button.addEventListener('click',()=>tab(i));
    button.addEventListener('keydown',event=>{
      if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){
        event.preventDefault();const next=event.key==='Home' ? 0 : event.key==='End' ? 1 : 1-i;tab(next);tabs[next].focus();
      }
    });
  });
  byId('topology-expand').addEventListener('click',()=>{
    if(dialog.open){dialog.close();return;}
    dialog.append(panel);dialog.showModal();
    text('topology-expand','还原视图');byId('topology-expand').focus();
  });
  dialog.addEventListener('close',()=>{anchor.after(panel);text('topology-expand','放大拓扑');byId('topology-expand').focus();});
  window.coolingTopology={
    reset(){model.reset();pinned=false;schedule();},
    start(source){model.start(source);pinned=false;schedule();},
    frame(frame){model.ingest(frame,now());schedule();},
    diagnosis(d){model.diagnose(d);schedule();},
    finish(status){model.finish(status);schedule();},
    preview(){model.reset('preview');pinned=false;schedule();},
    snapshot(){return model.snapshot(now());}
  };
  fetch('/static/topology.svg').then(response=>{
    if(!response.ok)throw new Error('拓扑图加载失败');return response.text();
  }).then(svg=>{drawing.innerHTML=svg;schedule();}).catch(()=>{
    drawing.textContent='拓扑图加载失败，请刷新页面。参数曲线与预警列表仍可查看。';
  });
  // Use the actual detector threshold when the service reports it.
  fetch('/api/v1/anomaly-model').then(r=>r.json()).then(r=>{
    if(Number.isFinite(r.data?.threshold)){model.threshold=r.data.threshold;schedule();}
  }).catch(()=>{});
  setInterval(schedule,1000);
  tab(0);render();
})();
