(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const panel = byId('qa-panel'), toggle = byId('qa-toggle');
  const messages = byId('qa-messages'), question = byId('qa-question');
  const suggestions = byId('qa-suggestions'), status = byId('qa-status');
  const defaults = ['监测传感器有哪些？', '当前是什么工况？', '查找3号阀的历史维修记录', '过滤器堵塞怎么处理？'];
  let history = [], busy = false;

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }
  function message(text, user = false, error = false) {
    const wrapper = element('article', null, `qa-message ${user ? 'qa-user' : ''} ${error ? 'qa-error' : ''}`);
    wrapper.append(element('div', user ? '你' : '知识问答助手', 'qa-speaker'));
    const body = element('div', null, 'qa-body');
    body.append(element('p', text));
    wrapper.append(body);
    messages.append(wrapper);
    while (messages.children.length > 40) messages.firstElementChild.remove();
    return body;
  }
  function showSuggestions(items) {
    suggestions.replaceChildren();
    for (const text of items.slice(0, 4)) {
      const button = element('button', text);
      button.type = 'button';
      button.disabled = busy;
      button.addEventListener('click', () => {
        question.value = text;
        byId('qa-form').requestSubmit();
      });
      suggestions.append(button);
    }
  }
  function render(data) {
    const body = message(data.answer);
    for (const card of data.cards || []) {
      if (card.type === 'table') {
        body.append(element('h3', card.title));
        const wrap = element('div', null, 'qa-table'), table = element('table');
        const head = element('thead'), headings = element('tr');
        for (const label of card.columns) {
          const cell = element('th', label);
          cell.scope = 'col';
          headings.append(cell);
        }
        head.append(headings); table.append(head);
        const rows = element('tbody');
        for (const row of card.rows) {
          const tr = element('tr');
          for (const value of row) tr.append(element('td', value));
          rows.append(tr);
        }
        table.append(rows); wrap.append(table); body.append(wrap);
      } else if (card.type === 'steps') {
        body.append(element('h3', card.title));
        const list = element('ol');
        for (const step of card.items) list.append(element('li', step));
        body.append(list);
        if (card.parts?.length) body.append(element('p', `备件参考：${card.parts.join('、')}`));
        if (card.notice) body.append(element('p', card.notice, 'qa-notice'));
      } else if (card.type === 'maintenance') {
        const order = element('section', null, 'qa-order');
        order.append(element('h3', `${card.date} · ${card.component}`));
        order.append(element('small', `${card.order_id} · ${card.source}`));
        order.append(element('p', card.action));
        if (card.note) order.append(element('p', card.note));
        body.append(order);
      }
    }
    if (data.sources?.length) {
      const sources = element('div', null, 'qa-source');
      sources.append(element('div', '来源'));
      for (const source of data.sources) {
        const line = element('div');
        // Only local API references become links; all answer content stays plain text.
        if (source.reference?.startsWith('/api/v1/')) {
          const link = element('a', source.title);
          link.href = source.reference; link.target = '_blank'; link.rel = 'noopener';
          line.append(link);
        } else {
          line.textContent = source.title;
          line.title = source.reference || '';
        }
        sources.append(line);
      }
      body.append(sources);
    }
    showSuggestions(data.suggestions || defaults);
  }
  function open(value) {
    panel.hidden = !value;
    toggle.setAttribute('aria-expanded', String(value));
    (value ? question : toggle).focus();
  }
  function clear() {
    history = [];
    messages.replaceChildren();
    question.value = '';
    for (const id of ['qa-keyword', 'qa-since', 'qa-until']) byId(id).value = '';
    message('你好，可以向我查询设备传感器、当前监测工况、维修记录和故障处置说明。每次回答都会重新查询，当前工况以本轮实际回放观测为准。');
    showSuggestions(defaults);
  }
  toggle.addEventListener('click', () => open(panel.hidden));
  byId('qa-close').addEventListener('click', () => open(false));
  panel.addEventListener('keydown', event => {
    if (event.key === 'Escape') open(false);
  });
  byId('qa-clear').addEventListener('click', clear);
  question.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      byId('qa-form').requestSubmit();
    }
  });
  byId('qa-form').addEventListener('submit', async event => {
    event.preventDefault();
    const text = question.value.trim();
    if (busy || !text) return;
    const since = byId('qa-since').value, until = byId('qa-until').value;
    if (since && until && since > until) {
      status.textContent = '起始日期不能晚于结束日期';
      return;
    }
    busy = true;
    for (const id of ['qa-send', 'qa-clear']) byId(id).disabled = true;
    for (const button of suggestions.children) button.disabled = true;
    question.value = '';
    message(text, true);
    messages.scrollTop = messages.scrollHeight;
    status.textContent = '正在查询…';
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch('/api/v1/agents/knowledge-qa/ask', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, signal: controller.signal,
        body: JSON.stringify({question: text, history: history.slice(-6),
          keyword: byId('qa-keyword').value.trim() || null, since: since || null, until: until || null})
      });
      const result = await response.json();
      if (!response.ok || result.code !== 0) {
        throw new Error(typeof result.detail === 'string' ? result.detail : '查询失败，请检查输入后重试');
      }
      history.push(text);
      history = history.slice(-6);
      render(result.data);
      status.textContent = '已查询 · 可继续追问';
    } catch (error) {
      message(error.name === 'AbortError' ? '查询超时，请稍后重试。' : error.message, false, true);
      if (!question.value) question.value = text;
      status.textContent = '查询未完成，请重试';
    } finally {
      clearTimeout(timer);
      busy = false;
      for (const id of ['qa-send', 'qa-clear']) byId(id).disabled = false;
      for (const button of suggestions.children) button.disabled = false;
      messages.scrollTop = messages.scrollHeight;
    }
  });
  clear();
})();
