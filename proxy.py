"""
LLM Proxy — OpenAI-compatible reverse proxy with request/response inspector
============================================================================
用法：
  pip install fastapi uvicorn httpx
  python proxy.py
  python proxy.py --upstream http://127.0.0.1:8008 --proxy-port 8000 --ui-port 8001 --max-records 200
  python proxy.py --think off   # 在每个请求体中注入 chat_template_kwargs.enable_thinking=false
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.responses import HTMLResponse


# ── 内联 HTML ─────────────────────────────────────────────────────────────────
HTML_CONTENT = r"""
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>LLM Proxy Inspector</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --bg:        #f5f4f0;
  --surface:   #ffffff;
  --border:    #e2e0d8;
  --border2:   #ccc9be;
  --text:      #1a1916;
  --text2:     #6b6860;
  --text3:     #9b9890;
  --accent:    #2a5bd7;
  --accent-bg: #eef2fd;
  --accent-bd: #c5d3f8;
  --green:     #1a7f4b;
  --green-bg:  #e6f6ed;
  --red:       #c0392b;
  --red-bg:    #fdecea;
  --yellow:    #92600a;
  --yellow-bg: #fef3dc;
  --mono:      'JetBrains Mono', monospace;
  --sans:      'DM Sans', sans-serif;
  --radius:    8px;
  --sidebar-w: 290px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  display: flex;
  height: 100vh;
  overflow: hidden;
  font-size: 13px;
}

/* ══ SIDEBAR ══════════════════════════════════════════════════════════════════ */

#sidebar {
  width: var(--sidebar-w);
  flex-shrink: 0;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

#sidebar-head {
  padding: 14px 14px 10px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

#sidebar-head h1 {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: .01em;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 6px;
}

#sidebar-head h1 .dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 2px var(--green-bg);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%,100% { opacity: 1; }
  50%      { opacity: .4; }
}

#sidebar-meta {
  margin-top: 5px;
  font-size: 11px;
  color: var(--text3);
  display: flex;
  gap: 10px;
}

#sidebar-list {
  flex: 1;
  overflow-y: auto;
  padding: 4px 0;
}

#sidebar-list::-webkit-scrollbar { width: 10px; }
#sidebar-list::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

.record-item {
  padding: 9px 14px;
  cursor: pointer;
  border-left: 3px solid transparent;
  border-bottom: 1px solid var(--bg);
  transition: background .1s;
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 4px 8px;
  align-items: start;
}
.record-item:hover  { background: var(--bg); }
.record-item.active { background: var(--accent-bg); border-left-color: var(--accent); }

.ri-method {
  font-family: var(--mono);
  font-size: 9px;
  font-weight: 500;
  padding: 2px 5px;
  border-radius: 3px;
  background: var(--accent-bg);
  color: var(--accent);
  border: 1px solid var(--accent-bd);
  align-self: center;
}
.ri-path {
  font-size: 11.5px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--text);
  align-self: center;
}
.ri-status {
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 500;
  align-self: center;
}
.ri-status.ok  { color: var(--green); }
.ri-status.err { color: var(--red); }
.ri-status.pending { color: var(--text3); }

.ri-meta {
  grid-column: 2 / -1;
  font-size: 10.5px;
  color: var(--text3);
  display: flex;
  gap: 8px;
  align-items: center;
}
.ri-badge {
  display: inline-block;
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 9.5px;
  font-weight: 500;
  background: var(--yellow-bg);
  color: var(--yellow);
  border: 1px solid #f0d9a0;
}
.ri-badge.sse { background: var(--green-bg); color: var(--green); border-color: #a8dfbe; }

/* ══ MAIN ══════════════════════════════════════════════════════════════════════ */

#main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0;
}

/* ── Toolbar ── */
#toolbar {
  height: 44px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 16px;
  gap: 10px;
  flex-shrink: 0;
}
#toolbar-path {
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--text2);
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
#toolbar-model {
  font-size: 11px;
  font-weight: 600;
  color: var(--accent);
  background: var(--accent-bg);
  border: 1px solid var(--accent-bd);
  padding: 2px 8px;
  border-radius: 99px;
}
.latency-badge {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text3);
  background: var(--bg);
  border: 1px solid var(--border);
  padding: 2px 7px;
  border-radius: 4px;
}
.stream-badge {
  font-size: 9.5px;
  font-weight: 600;
  padding: 2px 7px;
  border-radius: 4px;
  background: var(--green-bg);
  color: var(--green);
  border: 1px solid #a8dfbe;
}

/* ── Tab bar ── */
#tabbar {
  display: flex;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0 16px;
  gap: 0;
  flex-shrink: 0;
}
.tb {
  padding: 9px 14px;
  font-size: 12px;
  cursor: pointer;
  color: var(--text3);
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  user-select: none;
  transition: color .15s;
}
.tb:hover  { color: var(--text); }
.tb.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }

/* ── Content ── */
#content {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.panel { display: none; flex: 1; overflow: hidden; }
.panel.active { display: flex; flex-direction: column; }

/* Split panes for request/response */
#pane-split {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  flex: 1;
  overflow: hidden;
}

.pane {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-right: 1px solid var(--border);
}
.pane:last-child { border-right: none; }

.pane-head {
  padding: 8px 14px;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  font-size: 11px;
  font-weight: 600;
  color: var(--text2);
  letter-spacing: .05em;
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.pane-body {
  flex: 1;
  overflow-y: auto;
  padding: 14px;
}
.pane-body::-webkit-scrollbar { width: 10px; }
.pane-body::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* Code / pre */
pre, .json-pre {
  font-family: var(--mono);
  font-size: 12.5px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 14px;
}

/* Messages (pretty view) */
.msg-list { display: flex; flex-direction: column; gap: 10px; }

.msg-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}
.msg-card-head {
  padding: 6px 12px;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  font-size: 10px;
  font-weight: 600;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: .08em;
  display: flex;
  align-items: center;
  gap: 6px;
}
.role-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}
.role-dot.user      { background: var(--accent); }
.role-dot.assistant { background: var(--green); }
.role-dot.system    { background: var(--yellow); }

.msg-card-body {
  padding: 10px 12px;
  font-size: 12.5px;
  line-height: 1.75;
  white-space: pre-wrap;
  color: var(--text);
}

/* Thinking / reasoning block */
.reasoning-block {
  margin-top: 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}
.reasoning-toggle {
  padding: 6px 12px;
  background: var(--yellow-bg);
  border-bottom: 1px solid #f0d9a0;
  font-size: 10px;
  font-weight: 600;
  color: var(--yellow);
  cursor: pointer;
  user-select: none;
  display: flex;
  align-items: center;
  gap: 6px;
}
.reasoning-toggle::before { content: '▶'; font-size: 8px; transition: transform .2s; }
.reasoning-toggle.open::before { transform: rotate(90deg); }
.reasoning-body {
  display: none;
  padding: 10px 12px;
  font-size: 12.5px;
  line-height: 1.7;
  color: var(--text2);
  white-space: pre-wrap;
  font-family: var(--sans);
}
.reasoning-body.open { display: block; }

/* Tool calls block */
.tool-calls-block {
  margin-top: 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}
.tool-calls-toggle {
  padding: 6px 12px;
  background: var(--accent-bg);
  border-bottom: 1px solid var(--accent-bd);
  font-size: 10px;
  font-weight: 600;
  color: var(--accent);
  cursor: pointer;
  user-select: none;
  display: flex;
  align-items: center;
  gap: 6px;
}
.tool-calls-toggle::before { content: '▶'; font-size: 8px; transition: transform .2s; }
.tool-calls-toggle.open::before { transform: rotate(90deg); }
.tool-calls-body {
  display: none;
  padding: 8px 12px;
}
.tool-calls-body.open { display: block; }
.tool-call-item {
  margin-bottom: 8px;
  border: 1px solid var(--accent-bd);
  border-radius: 6px;
  overflow: hidden;
}
.tool-call-item:last-child { margin-bottom: 0; }
.tool-call-name {
  padding: 4px 10px;
  background: var(--accent-bg);
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--accent);
  border-bottom: 1px solid var(--accent-bd);
}
.tool-call-args {
  padding: 8px 10px;
  font-family: var(--mono);
  font-size: 11.5px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text);
  background: var(--surface);
}

/* Tool result block (role=tool) */
.tool-result-id {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text3);
  margin-bottom: 4px;
}

/* SSE chunks raw view */
#chunks-view {
  padding: 14px;
  overflow-y: auto;
  flex: 1;
}
.chunk-line {
  font-family: var(--mono);
  font-size: 11.5px;
  line-height: 1.6;
  color: var(--text2);
  border-bottom: 1px solid var(--bg);
  padding: 2px 0;
}
.chunk-line.data   { color: var(--text); }
.chunk-line.done   { color: var(--green); font-weight: 500; }
.chunk-line.empty  { opacity: .3; }

/* Empty / loading states */
#empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 8px;
  color: var(--text3);
}
#empty .big { font-size: 28px; }
#empty p { font-size: 13px; }

/* Clear button */
#clear-btn {
  margin-left: auto;
  font-size: 10px;
  font-weight: 500;
  padding: 2px 8px;
  border: 1px solid var(--border2);
  border-radius: 4px;
  background: var(--surface);
  color: var(--text3);
  cursor: pointer;
  font-family: var(--sans);
}
#clear-btn:hover { background: var(--red-bg); color: var(--red); border-color: #e8b4b0; }

/* Copy button */
.copy-btn {
  margin-left: auto;
  font-size: 10px;
  font-weight: 500;
  padding: 2px 8px;
  border: 1px solid var(--border2);
  border-radius: 4px;
  background: var(--surface);
  color: var(--text2);
  cursor: pointer;
  font-family: var(--sans);
}
.copy-btn:hover { background: var(--bg); }
</style>
</head>
<body>

<!-- ══ SIDEBAR ══════════════════════════════════════════════════════════════ -->
<div id="sidebar">
  <div id="sidebar-head">
    <h1><span class="dot"></span> LLM Proxy Inspector</h1>
    <div id="sidebar-meta">
      <span id="count-badge">0 条记录</span>
      <span id="upstream-badge"></span>
      <button id="clear-btn" onclick="clearHistory()">清除</button>
    </div>
  </div>
  <div id="sidebar-list">
    <div id="list-empty" style="padding:20px 14px;color:var(--text3);font-size:12px">
      等待请求…
    </div>
  </div>
</div>

<!-- ══ MAIN ════════════════════════════════════════════════════════════════ -->
<div id="main">
  <!-- 未选中状态 -->
  <div id="empty" style="display:flex">
    <div class="big">🔍</div>
    <p>从左侧选择一条记录</p>
  </div>

  <!-- 选中后的内容区 -->
  <div id="detail" style="display:none;flex:1;flex-direction:column;overflow:hidden;">

    <div id="toolbar">
      <span id="toolbar-path">—</span>
      <span id="toolbar-model" style="display:none"></span>
      <span id="toolbar-latency" class="latency-badge" style="display:none"></span>
      <span id="toolbar-stream" class="stream-badge" style="display:none">SSE Stream</span>
    </div>

    <div id="tabbar">
      <div class="tb active" data-panel="pretty" onclick="switchTab(this)">消息</div>
      <div class="tb" data-panel="raw-json" onclick="switchTab(this)">Raw JSON</div>
    </div>

    <div id="content">

      <!-- ── Pretty 消息视图 ── -->
      <div id="panel-pretty" class="panel active">
        <div id="pane-split">
          <!-- 请求 -->
          <div class="pane">
            <div class="pane-head">
              ↑ Request
              <button class="copy-btn" onclick="copyPaneJson('req', this)">复制</button>
            </div>
            <div class="pane-body">
              <div id="req-messages" class="msg-list"></div>
            </div>
          </div>
          <!-- 响应 -->
          <div class="pane">
            <div class="pane-head">
              ↓ Response
              <button class="copy-btn" onclick="copyPaneJson('resp', this)">复制</button>
            </div>
            <div class="pane-body">
              <div id="resp-messages" class="msg-list"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- ── Raw JSON ── -->
      <div id="panel-raw-json" class="panel">
        <div id="pane-split" style="display:grid;grid-template-columns:1fr 1fr;flex:1;overflow:hidden">
          <div class="pane">
            <div class="pane-head">↑ Request JSON
              <button class="copy-btn" onclick="copyText(document.getElementById('req-raw').textContent, this)">复制</button>
            </div>
            <div class="pane-body"><pre id="req-raw"></pre></div>
          </div>
          <div class="pane">
            <div class="pane-head">↓ Response JSON
              <button class="copy-btn" onclick="copyText(document.getElementById('resp-raw').textContent, this)">复制</button>
            </div>
            <div class="pane-body"><pre id="resp-raw"></pre></div>
          </div>
        </div>
      </div>

    </div><!-- #content -->
  </div><!-- #detail -->
</div><!-- #main -->

<script>
// ══ State ══════════════════════════════════════════════════════════════════
let currentId = null;
let currentRecord = null;
let records = [];   // sidebar list cache

// Read id from URL path /ids/<id>
function idFromUrl() {
  const m = location.pathname.match(/^\/ids\/(.+)/);
  return m ? m[1] : null;
}

// ══ Sidebar refresh (only replaces list HTML) ═══════════════════════════════
async function refreshSidebar() {
  try {
    const res = await fetch('/api/records');
    records = await res.json();
  } catch { return; }

  document.getElementById('count-badge').textContent = records.length + ' 条记录';

  const list = document.getElementById('sidebar-list');
  if (!records.length) {
    list.innerHTML = '<div id="list-empty" style="padding:20px 14px;color:var(--text3);font-size:12px">等待请求…</div>';
    return;
  }

  list.innerHTML = records.map(r => {
    const active   = r.id === currentId ? 'active' : '';
    const statusCls = r.status == null ? 'pending' : (r.status < 400 ? 'ok' : 'err');
    const statusTxt = r.status == null ? '…' : r.status;
    const latency   = r.latency != null ? r.latency + 'ms' : '';
    const badge     = r.is_sse
      ? '<span class="ri-badge sse">SSE</span>'
      : (r.stream ? '<span class="ri-badge">stream</span>' : '');
    const model = r.model ? `<span style="font-size:10px;color:var(--accent)">${esc(r.model.split('/').pop())}</span>` : '';
    return `
<div class="record-item ${active}" onclick="selectRecord('${r.id}')">
  <span class="ri-method">${r.method}</span>
  <span class="ri-path">${esc(r.path)}</span>
  <span class="ri-status ${statusCls}">${statusTxt}</span>
  <div class="ri-meta">${model}${badge}<span>${esc(r.time)}${latency ? ' · ' + latency : ''}</span></div>
</div>`;
  }).join('');
}

// ══ Select & load detail ══════════════════════════════════════════════════
async function selectRecord(id) {
  currentId = id;
  history.replaceState(null, '', '/ids/' + id);

  await refreshSidebar();

  try {
    const res = await fetch('/api/records/' + id);
    currentRecord = await res.json();
  } catch { return; }

  renderDetail(currentRecord);
}

// ══ Render detail ══════════════════════════════════════════════════════════
function renderDetail(r) {
  document.getElementById('empty').style.display  = 'none';
  document.getElementById('detail').style.display = 'flex';

  // Toolbar
  document.getElementById('toolbar-path').textContent = r.method + ' ' + r.path;

  const modelEl = document.getElementById('toolbar-model');
  if (r.model) { modelEl.textContent = r.model.split('/').pop(); modelEl.style.display = ''; }
  else modelEl.style.display = 'none';

  const latEl = document.getElementById('toolbar-latency');
  if (r.latency != null) { latEl.textContent = r.latency + ' ms'; latEl.style.display = ''; }
  else latEl.style.display = 'none';

  const streamEl = document.getElementById('toolbar-stream');
  streamEl.style.display = r.is_sse ? '' : 'none';

  // ── Pretty: Request messages ──
  const reqMsg = document.getElementById('req-messages');
  const reqJson = r.req_json || {};
  const msgs = reqJson.messages || [];
  reqMsg.innerHTML = msgs.length
    ? msgs.map(m => msgCard(m.role, m.content, null, m.tool_calls)).join('')
    : `<pre>${esc(JSON.stringify(reqJson, null, 2))}</pre>`;

  // ── Pretty: Response messages ──
  const respMsg = document.getElementById('resp-messages');
  if (r.is_sse && r.resp_merged) {
    const choices = r.resp_merged.choices || [];
    respMsg.innerHTML = choices.map(c => {
      // 泛化合并后 delta 字段直接在 choice 上，无 message 包装
      return msgCard(c.role || 'assistant', c.content, c.reasoning_content, c.tool_calls);
    }).join('') || '<span style="color:var(--text3);font-size:12px">流式响应解析中…</span>';
  } else if (r.resp_json) {
    const choices = r.resp_json.choices || [];
    respMsg.innerHTML = choices.length
      ? choices.map(c => {
          const msg = c.message || {};
          return msgCard(msg.role || 'assistant', msg.content, msg.reasoning_content, msg.tool_calls);
        }).join('')
      : `<pre>${esc(JSON.stringify(r.resp_json, null, 2))}</pre>`;
  } else if (r.resp_raw) {
    respMsg.innerHTML = `<pre>${esc(r.resp_raw)}</pre>`;
  } else {
    respMsg.innerHTML = '<span style="color:var(--text3);font-size:12px">等待响应…</span>';
  }

  // ── Raw JSON ──
  document.getElementById('req-raw').textContent  = JSON.stringify(r.req_json, null, 2) || '(empty)';
  const respData = r.is_sse ? r.resp_merged : r.resp_json;
  document.getElementById('resp-raw').textContent = JSON.stringify(respData, null, 2) || r.resp_raw || '(pending)';

  // keep current tab if valid
  const activeTab = document.querySelector('.tb.active');
  if (activeTab) switchTab(activeTab, true);
}

// ══ Helpers ════════════════════════════════════════════════════════════════

function msgCard(role, content, reasoning, toolCalls) {
  const dotCls = ['user','assistant','system','tool'].includes(role) ? role : 'user';

  // For role=tool, content is the tool result
  let bodyHtml = '';
  if (role === 'tool') {
    // content may be array of objects or plain string
    const parts = Array.isArray(content) ? content : null;
    const text = parts
      ? parts.map(p => p.text ?? JSON.stringify(p)).join('\n')
      : (content || '');
    bodyHtml = `<div class="msg-card-body">${esc(text)}</div>`;
  } else if (Array.isArray(content)) {
    // content blocks array (e.g. [{type:'text', text:'...'}, {type:'image',...}])
    bodyHtml = content.map(block => {
      if (block.type === 'text') return `<div class="msg-card-body">${esc(block.text || '')}</div>`;
      return `<div class="msg-card-body" style="color:var(--text2);font-size:11px">[${esc(block.type || 'block')}]</div>`;
    }).join('');
  } else {
    bodyHtml = `<div class="msg-card-body">${esc(content || '')}</div>`;
  }

  let html = `
<div class="msg-card">
  <div class="msg-card-head">
    <span class="role-dot ${dotCls}"></span>${esc(role)}
  </div>
  ${bodyHtml}`;

  if (reasoning) {
    html += `
  <div class="reasoning-block">
    <div class="reasoning-toggle open" onclick="toggleBlock(this)">思考链</div>
    <div class="reasoning-body open">${esc(reasoning)}</div>
  </div>`;
  }

  if (toolCalls && toolCalls.length) {
    const itemsHtml = toolCalls.map(tc => {
      const fn = tc.function || {};
      const name = fn.name || tc.name || tc.id || '(unknown)';
      let args = fn.arguments ?? tc.arguments ?? '';
      try { args = JSON.stringify(JSON.parse(args), null, 2); } catch {}
      return `
    <div class="tool-call-item">
      <div class="tool-call-name">⚙ ${esc(name)}</div>
      <div class="tool-call-args">${esc(args)}</div>
    </div>`;
    }).join('');
    html += `
  <div class="tool-calls-block">
    <div class="tool-calls-toggle open" onclick="toggleBlock(this)">Tool Calls (${toolCalls.length})</div>
    <div class="tool-calls-body open">${itemsHtml}</div>
  </div>`;
  }

  html += '</div>';
  return html;
}

function toggleBlock(el) {
  el.classList.toggle('open');
  el.nextElementSibling.classList.toggle('open');
}

function switchTab(el, silent) {
  document.querySelectorAll('.tb').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  const panelId = 'panel-' + el.dataset.panel;
  const panel = document.getElementById(panelId);
  if (panel) panel.classList.add('active');
}

function esc(s) {
  return String(s || '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function copyText(text, btn) {
  const feedback = (ok) => {
    if (!btn) return;
    const orig = btn.textContent;
    btn.textContent = ok ? '已复制' : '失败';
    btn.style.color = ok ? 'var(--accent, #4a9eff)' : '#e55';
    setTimeout(() => { btn.textContent = orig; btn.style.color = ''; }, 1500);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => feedback(true), () => fallback(text, btn, feedback));
  } else {
    fallback(text, btn, feedback);
  }
}

function fallback(text, btn, feedback) {
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
    document.body.appendChild(ta);
    ta.focus(); ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    if (feedback) feedback(ok);
  } catch(e) {
    if (feedback) feedback(false);
  }
}

function copyPaneJson(side, btn) {
  if (!currentRecord) return;
  const data = side === 'req'
    ? currentRecord.req_json
    : (currentRecord.is_sse ? currentRecord.resp_merged : currentRecord.resp_json);
  copyText(JSON.stringify(data, null, 2), btn);
}

// ══ Clear history ═════════════════════════════════════════════════════════
async function clearHistory() {
  try {
    await fetch('/api/records', { method: 'DELETE' });
  } catch { return; }
  currentId = null;
  currentRecord = null;
  history.replaceState(null, '', '/');
  document.getElementById('empty').style.display = 'flex';
  document.getElementById('detail').style.display = 'none';
  await refreshSidebar();
}

// ══ Boot ══════════════════════════════════════════════════════════════════
(async () => {
  await refreshSidebar();
  const initId = idFromUrl();
  if (initId) selectRecord(initId);
  setInterval(refreshSidebar, 5000);
})();
</script>
</body>
</html>
"""

# ── 配置 ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="LLM Proxy Inspector")
_parser.add_argument("--upstream",    default=os.getenv("UPSTREAM_BASE", "http://127.0.0.1:8008"))
_parser.add_argument("--proxy-port",  type=int, default=int(os.getenv("PROXY_PORT", "8000")))
_parser.add_argument("--ui-port",     type=int, default=int(os.getenv("UI_PORT",    "8001")))
_parser.add_argument("--max-records", type=int, default=int(os.getenv("MAX_RECORDS","200")))
_parser.add_argument("--think",       choices=["on", "off"], default="on",
                     help="当为 off 时在请求体中注入 chat_template_kwargs.enable_thinking=false")
_parser.add_argument("--params",      default=None, metavar="JSON",
                     help='启动时注入到每个请求体的参数，JSON 格式，如 \'{"temperature":0.7,"top_p":0.9}\'')
_args = _parser.parse_args()

UPSTREAM_BASE = _args.upstream.rstrip("/")
PROXY_PORT    = _args.proxy_port
UI_PORT       = _args.ui_port
MAX_RECORDS   = _args.max_records
THINK         = _args.think  # "on" / "off" / None

# ── 可在运行时动态修改的请求参数覆盖 ──────────────────────────────────────────
try:
    _override_params: dict = json.loads(_args.params) if _args.params else {}
    if not isinstance(_override_params, dict):
        raise ValueError("--params 必须是 JSON 对象")
except Exception as e:
    print(f"[warn] --params 解析失败：{e}，已忽略")
    _override_params = {}
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("proxy")

_store: OrderedDict[str, dict] = OrderedDict()
_lock  = asyncio.Lock()


# ── SSE 解析 ──────────────────────────────────────────────────────────────────

def _merge_delta(acc: dict, delta: dict) -> None:
    """递归合并一个 delta 片段到累积 dict。

    Pure CPU operation — safe to call from async context without blocking the event loop.

    规则：
      - None 值跳过，不覆盖已有内容
      - str（增量字段）→ 拼接，仅限 content / reasoning_content / arguments
      - str（其他字段）→ 覆盖，如 type / role / name / id 等只出现一次的字段
      - list → 按元素的 "index" 字段找到同位置条目后递归合并（tool_calls 场景）
      - dict → 递归合并（function: {name, arguments} 等嵌套结构）
      - 其他 → 直接覆盖（finish_reason / logprobs 等）
    """
    for key, val in delta.items():
        if val is None:
            continue
        if key not in acc or acc[key] is None:
            # 首次出现：深拷贝避免与原始 chunk 共享引用
            acc[key] = copy.deepcopy(val)
        elif isinstance(val, str) and isinstance(acc[key], str) and key in ("content", "reasoning_content", "arguments"):
            # 增量文本拼接，如 content / reasoning_content / arguments
            acc[key] += val
        elif isinstance(val, list) and isinstance(acc[key], list):
            # 列表元素按 "index" 字段定位后递归合并
            # 典型场景：tool_calls，每个 chunk 携带部分 arguments
            for item in val:
                if not isinstance(item, dict):
                    acc[key].append(item)
                    continue
                item_idx = item.get("index")
                existing = next(
                    (x for x in acc[key] if isinstance(x, dict) and x.get("index") == item_idx),
                    None,
                ) if item_idx is not None else None
                if existing is None:
                    # 新的 tool_call 条目，首次出现
                    acc[key].append(copy.deepcopy(item))
                else:
                    _merge_delta(existing, item)
        elif isinstance(val, dict) and isinstance(acc[key], dict):
            # 嵌套 dict 递归合并，如 function: {name, arguments}
            _merge_delta(acc[key], val)
        else:
            # 覆盖：role 等只出现一次的字段，或类型与首次不同时（上游异常场景）
            if type(acc[key]) is not type(val):
                log.warning("merge type mismatch key=%r acc=%r val=%r, overwriting", key, type(acc[key]).__name__, type(val).__name__)
            acc[key] = val


async def _save(record_id: str, data: dict):
    async with _lock:
        _store[record_id] = data
        while len(_store) > MAX_RECORDS:
            _store.popitem(last=False)

async def _save_merge(record_id: str, data: dict):
    """流式结束后异步更新记录（增量合并，不存储原始 SSE 行）。"""
    async with _lock:
        if record_id in _store:
            _store[record_id].update(data)
        else:
            log.debug("_save_merge: record %s already evicted", record_id)


async def _update_record(rid: str, status: int, latency: int, merged: dict):
    """流式结束后更新记录。"""
    await _save_merge(rid, {
        "status": status,
        "latency": latency,
        "is_sse": True,
        "resp_merged": merged,
    })


def _log_update_error(task: asyncio.Task) -> None:
    """回调：记录 _update_record 任务的异常。"""
    try:
        exc = task.exception()
        if exc:
            log.error("update_record error: %s", exc)
    except Exception:
        pass


# ── Proxy App ─────────────────────────────────────────────────────────────────

proxy_app = FastAPI(title="LLM Proxy")


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

REQUEST_HEADERS_TO_DROP = HOP_BY_HOP_HEADERS | {
    "host",
    "content-length",
}

RESPONSE_HEADERS_TO_DROP = HOP_BY_HOP_HEADERS | {
    "content-encoding",
    "content-length",
}


@proxy_app.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE","PATCH","OPTIONS"])
async def proxy(path: str, request: Request):
    url = f"{UPSTREAM_BASE}/{path}"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in REQUEST_HEADERS_TO_DROP
    }
    headers["accept-encoding"] = "identity"

    body_bytes = await request.body()
    try:
        req_json = json.loads(body_bytes) if body_bytes else None
    except Exception:
        req_json = None

    req_modified = False

    # 当 --think off 时，向请求体注入 chat_template_kwargs.enable_thinking=false
    if THINK == "off" and isinstance(req_json, dict):
        ktw = req_json.get("chat_template_kwargs")
        if not isinstance(ktw, dict):
            ktw = {}
        ktw["enable_thinking"] = False
        req_json["chat_template_kwargs"] = ktw
        req_modified = True

    # 注入参数覆盖（来自 --params，启动时设定，只读）
    if isinstance(req_json, dict) and _override_params:
        for key, val in _override_params.items():
            req_json[key] = val
        req_modified = True

    if req_modified:
        body_bytes = json.dumps(req_json, ensure_ascii=False).encode()

    is_stream = isinstance(req_json, dict) and req_json.get("stream", False)
    record_id = str(uuid.uuid4())
    ts = datetime.now().strftime("%H:%M:%S")
    model = req_json.get("model", "") if req_json else ""

    # 基础记录（先存，流结束后补充）
    base_record = {
        "id":       record_id,
        "time":     ts,
        "method":   request.method,
        "path":     f"/{path}",
        "model":    model,
        "stream":   is_stream,
        "req_json": req_json,
        "status":   None,
        "latency":  None,
        "is_sse":   False,
        "resp_json":   None,   # 非流式
        "resp_merged": None,   # 流式合并后
        "resp_raw":    None,   # 原始文本（非流式）
    }
    await _save(record_id, base_record)

    t0 = time.monotonic()
    log.info("→ %s %s  model=%s  stream=%s", request.method, url, model or "-", is_stream)

    # ── 流式 ──────────────────────────────────────────────
    if is_stream:
        # 增量解析状态：闭包累积，不收集原始 SSE 行节省内存
        merge_state: dict = {
            "id": None, "object": "chat.completion", "created": None, "model": None,
            "usage": None, "_choices": {},
        }
        # 解析队列：转发协程写入原始行，解析协程消费
        # 队列容量：平衡内存与数据完整性（50000 ≈ 50 万 token）
        parse_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=50000)
        parsed_chunk_count: int = 0  # 解析协程内部计数
        dropped_chunk_count: int = 0  # 丢弃的 chunk 数

        async def _parse_worker() -> int:
            """后台解析协程：从队列读取并解析 SSE，不阻塞转发路径。
            返回实际解析的 chunk 数量。"""
            nonlocal parsed_chunk_count
            try:
                while True:
                    raw_line = await parse_queue.get()
                    if raw_line is None:  # 结束信号
                        return parsed_chunk_count
                    parsed_chunk_count += 1
                    # 增量解析 SSE
                    if raw_line.startswith("data:"):
                        payload = raw_line[5:].strip()
                        if payload != "[DONE]":
                            try:
                                chunk = json.loads(payload)
                                if merge_state["id"] is None:
                                    merge_state["id"] = chunk.get("id")
                                    merge_state["created"] = chunk.get("created")
                                    merge_state["model"] = chunk.get("model")
                                if chunk.get("usage"):
                                    merge_state["usage"] = chunk["usage"]
                                for choice in chunk.get("choices", []):
                                    idx = choice.get("index", 0)
                                    if idx not in merge_state["_choices"]:
                                        merge_state["_choices"][idx] = {"index": idx, "finish_reason": None}
                                    _merge_delta(merge_state["_choices"][idx], choice.get("delta", {}))
                                    top = {k: v for k, v in choice.items() if k not in ("delta", "index")}
                                    if top:
                                        _merge_delta(merge_state["_choices"][idx], top)
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                log.error("_parse_worker crashed: %s", e)
                return parsed_chunk_count

        # 先建立连接获取状态码
        stream_client = httpx.AsyncClient(timeout=300)
        try:
            req = stream_client.build_request(
                request.method, url,
                headers=headers, content=body_bytes
            )
            upstream = await stream_client.send(req, stream=True)
        except Exception:
            await stream_client.aclose()
            raise
        status_code = upstream.status_code
        log.info("← %s %s  status=%s  [SSE started]", request.method, url, status_code)

        async def stream_gen() -> AsyncIterator[bytes]:
            nonlocal dropped_chunk_count
            # 启动后台解析协程
            parse_task = asyncio.create_task(_parse_worker())

            try:
                async for raw_line in upstream.aiter_lines():
                    # 空行也转发（SSE 事件边界）
                    yield (raw_line + "\n").encode()

                    # 非空行才放入解析队列
                    stripped = raw_line.strip()
                    if stripped:
                        try:
                            parse_queue.put_nowait(stripped)
                        except asyncio.QueueFull:
                            try:
                                await asyncio.wait_for(
                                    parse_queue.put(stripped), timeout=5.0
                                )
                            except asyncio.TimeoutError:
                                dropped_chunk_count += 1
                                log.warning(
                                    "parse queue timeout after 5s, dropping chunk"
                                )
            except Exception as exc:
                log.error("✗ %s %s  error=%s", request.method, url, exc)
                yield f"data: [ERROR] {exc}\n\n".encode()
            finally:
                # 关闭上游连接
                await upstream.aclose()
                await stream_client.aclose()

                # 发送结束信号：队列满时阻塞等待（parse_worker 会消费空位）
                await parse_queue.put(None)
                chunk_count = await parse_task

                latency = round((time.monotonic() - t0) * 1000)
                merged = {
                    "id": merge_state["id"],
                    "object": merge_state["object"],
                    "created": merge_state["created"],
                    "model": merge_state["model"],
                    "usage": merge_state["usage"],
                    "choices": [merge_state["_choices"][i] for i in sorted(merge_state["_choices"])],
                }
                if dropped_chunk_count > 0:
                    log.warning("← %s %s  status=%s  %dms  [SSE done, chunks=%d, dropped=%d]",
                                request.method, url, status_code, latency, chunk_count, dropped_chunk_count)
                else:
                    log.info("← %s %s  status=%s  %dms  [SSE done, chunks=%d]",
                             request.method, url, status_code, latency, chunk_count)
                task = asyncio.create_task(_update_record(record_id, status_code, latency, merged))
                task.add_done_callback(_log_update_error)

        return StreamingResponse(
            stream_gen(),
            media_type="text/event-stream",
            status_code=status_code,
            headers={"X-Proxy-Record-Id": record_id},
        )

    # ── 非流式 ────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            resp = await client.request(
                request.method, url,
                headers=headers, content=body_bytes
            )
        except Exception as exc:
            log.error("✗ %s %s  error=%s", request.method, url, exc)
            raise
        latency = round((time.monotonic() - t0) * 1000)
        log.info("← %s %s  status=%s  %dms", request.method, url, resp.status_code, latency)
        try:
            resp_json = resp.json()
        except Exception:
            resp_json = None

        async with _lock:
            if record_id in _store:
                _store[record_id].update({
                    "status":   resp.status_code,
                    "latency":  latency,
                    "is_sse":   False,
                    "resp_json": resp_json,
                    "resp_raw":  resp.text if resp_json is None else None,
                })

        response_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in RESPONSE_HEADERS_TO_DROP
        }

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={
                **response_headers,
                "X-Proxy-Record-Id": record_id,
            },
            media_type=resp.headers.get("content-type"),
        )


# ── UI App ────────────────────────────────────────────────────────────────────

ui_app = FastAPI(title="LLM Proxy UI")


@ui_app.get("/api/records")
async def api_records():
    """侧边栏列表，轻量数据。"""
    async with _lock:
        items = list(_store.values())
    result = []
    for r in reversed(items):
        result.append({
            "id":      r["id"],
            "time":    r["time"],
            "method":  r["method"],
            "path":    r["path"],
            "model":   r.get("model", ""),
            "stream":  r.get("stream", False),
            "status":  r.get("status"),
            "latency": r.get("latency"),
            "is_sse":  r.get("is_sse", False),
            "done":    r.get("status") is not None,
        })
    return result


@ui_app.delete("/api/records")
async def api_clear_records():
    """清除所有历史记录。使用 _lock 保证与写入互斥。"""
    async with _lock:
        _store.clear()
    return {"cleared": True}


@ui_app.get("/api/records/{record_id}")
async def api_record_detail(record_id: str):
    """单条记录完整数据。"""
    async with _lock:
        r = _store.get(record_id)
    if not r:
        return JSONResponse({"error": "not found"}, status_code=404)
    return r


@ui_app.get("/ids/{record_id}")
@ui_app.get("/")
def index():
    return HTMLResponse(HTML_CONTENT)




# ── 启动两个服务 ──────────────────────────────────────────────────────────────

async def main():
    cfg_proxy = uvicorn.Config(proxy_app, host="0.0.0.0", port=PROXY_PORT, log_level="warning")
    cfg_ui    = uvicorn.Config(ui_app,    host="0.0.0.0", port=UI_PORT,    log_level="warning")

    print(f"Proxy  → http://0.0.0.0:{PROXY_PORT}  (upstream: {UPSTREAM_BASE})")
    print(f"UI     → http://0.0.0.0:{UI_PORT}")
    print(f"think  → {THINK}")
    if _override_params:
        print(f"params → {json.dumps(_override_params, ensure_ascii=False)}")
    else:
        print(f"params → (none)")
    print()

    await asyncio.gather(
        uvicorn.Server(cfg_proxy).serve(),
        uvicorn.Server(cfg_ui).serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
