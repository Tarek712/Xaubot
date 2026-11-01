import asyncio, json, time, numpy as np, httpx
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()
candles, clients, last_p, spread = deque(maxlen=200), [], 3800.0, 0.25
state = {
    "bal": 10000.0, "eq": 10000.0, "pos": None, "win": 0, "tot": 0,
    "h": 0.5, "ent": 0.5, "reg": "CALIBRATING", "zones": [], "logs": []
}

def log(msg):
    state["logs"].insert(0, f"[{time.strftime('%H:%M:%S')}] {msg}")
    state["logs"] = state["logs"][:20]

def analyze():
    global last_p
    if len(candles) < 30: return
    closes = np.array([c["close"] for c in candles], dtype=float)
    rets = np.diff(closes)
    # 1. Hurst Exponent & Shannon Entropy (Autonomous Market Physics)
    std_diff = np.std(np.subtract(closes[4:], closes[:-4]))
    state["h"] = round(float(np.clip(std_diff / (np.std(rets) * 2.0 + 1e-6), 0.2, 0.85)), 2)
    hist, _ = np.histogram(rets[-40:], bins=6, density=True)
    p = hist[hist > 0] / max(1e-6, np.sum(hist[hist > 0]))
    state["ent"] = round(float(-np.sum(p * np.log2(p)) / np.log2(6)), 2)
    state["reg"] = "PERSISTENT" if state["h"] > 0.52 else ("MEAN_REV" if state["h"] < 0.48 else "RANDOM_NOISE")
    
    # 2. Price Gravity Zones (Unsupervised Clustering)
    grid = np.linspace(min(closes), max(closes), 8)
    state["zones"] = [round(float(x), 2) for x in grid[np.argsort([np.sum(np.exp(-0.5*((closes-g)/1.2)**2)) for g in grid])[-3:]]]
    
    # 3. Autonomous Execution & Position Management
    pos = state["pos"]
    dyn_risk = max(1.2, float(np.std(rets[-15:])) * 2.2)
    if pos:
        diff = (last_p - pos["entry"]) if pos["side"] == "BUY" else (pos["entry"] - last_p)
        pos["pnl"] = round(diff * pos["lots"] * 100.0, 2)
        state["eq"] = round(state["bal"] + pos["pnl"], 2)
        if not pos["be"] and diff >= dyn_risk * 0.7:
            pos["be"] = True
            pos["sl"] = pos["entry"]
            log(f"تأمين الصفقة #{pos['id']} ونقل الوقف لنقطة الدخول (Break-Even)")
        if (pos["side"] == "BUY" and (last_p >= pos["tp"] or last_p <= pos["sl"])) or \
           (pos["side"] == "SELL" and (last_p <= pos["tp"] or last_p >= pos["sl"])):
            state["bal"] = round(state["bal"] + pos["pnl"], 2)
            state["win"] += 1 if pos["pnl"] >= 0 else 0
            state["tot"] += 1
            log(f"إغلاق صفقة {pos['side']} عند {last_p:.2f}$ | الصافي: {pos['pnl']:+.2f}$")
            state["pos"] = None
    elif state["ent"] < 0.85 and state["reg"] != "RANDOM_NOISE":
        drift = float(np.mean(rets[-5:]))
        if abs(drift) > dyn_risk * 0.15:
            side = "BUY" if drift > 0 else "SELL"
            state["pos"] = {
                "id": int(time.time()) % 10000, "side": side, "entry": last_p,
                "sl": round(last_p - dyn_risk if side == "BUY" else last_p + dyn_risk, 2),
                "tp": round(last_p + dyn_risk * 1.8 if side == "BUY" else last_p - dyn_risk * 1.8, 2),
                "lots": 0.25, "pnl": 0.0, "be": False
            }
            log(f"دخول مستقل [{side}] عند {last_p:.2f}$ | H={state['h']} | إنتروبيا={state['ent']}")

async def market_loop():
    global last_p, spread
    async with httpx.AsyncClient(timeout=4.0) as c:
        try:
            r = await c.get("https://api.binance.com/api/v3/klines?symbol=PAXGUSDT&interval=1m&limit=70")
            for k in r.json():
                candles.append({"time": int(k[0]//1000), "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4])})
            last_p = candles[-1]["close"]
        except Exception: pass
    if not candles:
        t0 = int(time.time()) - 3600
        for i in range(60): candles.append({"time": t0 + i*60, "open": 3800.0, "high": 3802.0, "low": 3798.0, "close": 3800.5})

    while True:
        try:
            last_p = round(last_p + np.random.normal(0, 0.45), 2)
            spread = round(float(np.random.uniform(0.15, 0.35)), 2)
            c = candles[-1]
            now_m = (int(time.time()) // 60) * 60
            if now_m > c["time"]:
                candles.append({"time": now_m, "open": last_p, "high": last_p, "low": last_p, "close": last_p})
            else:
                c["close"] = last_p
                c["high"], c["low"] = max(c["high"], last_p), min(c["low"], last_p)
            analyze()
            for ws in list(clients):
                try: await ws.send_json({"price": last_p, "candle": candles[-1], "state": state, "spread": spread})
                except Exception: clients.remove(ws)
        except Exception as e: print("Loop err:", e)
        await asyncio.sleep(1.0)

@app.on_event("startup")
async def start(): asyncio.create_task(market_loop())

@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    await ws.send_json({"init": list(candles), "state": state, "price": last_p})
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        if ws in clients: clients.remove(ws)

HTML = """<!DOCTYPE html><html lang="ar" dir="rtl"><head><meta charset="UTF-8"><title>XAU/USD Bot</title>
<script src="https://cdn.tailwindcss.com"></script><script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<body class="bg-[#0B0E14] text-slate-200 p-3 font-sans select-none">
<div class="flex justify-between items-center p-3 bg-[#121722] rounded-xl border border-slate-800 mb-3">
  <div class="flex items-center gap-3">
    <span class="w-3 h-3 bg-emerald-400 rounded-full animate-pulse"></span>
    <span class="font-mono font-bold text-amber-400 text-lg">XAU/USD</span>
    <span id="price" class="font-mono text-2xl font-bold text-white">--.--</span>
    <span id="spread" class="text-xs text-slate-400 font-mono">--</span>
  </div>
  <div class="flex gap-6 text-xs font-mono">
    <div>رصيد: <b id="bal" class="text-white">$10,000</b></div>
    <div>الملكية: <b id="eq" class="text-emerald-400">$10,000</b></div>
    <div>النجاح: <b id="wr" class="text-amber-400">0%</b></div>
    <div>هيرست: <b id="hurst" class="text-sky-400">0.5</b></div>
  </div>
</div>
<div class="grid grid-cols-1 lg:grid-cols-3 gap-3">
  <div class="lg:col-span-2 bg-[#121722] p-3 rounded-xl border border-slate-800 h-[480px]" id="ch"></div>
  <div class="bg-[#121722] p-3 rounded-xl border border-slate-800 flex flex-col h-[480px]">
    <h3 class="text-xs font-bold text-amber-400 mb-2">سجل قرارات الوكيل المستقل</h3>
    <div id="pos-info" class="p-2 mb-2 bg-slate-900 rounded border border-slate-800 text-xs text-slate-400 font-mono">لا توجد صفقة مفتوحة</div>
    <div id="logs" class="flex-1 overflow-y-auto space-y-1.5 text-xs text-slate-300 font-mono"></div>
  </div>
</div>
<script>
const cEl = document.getElementById('ch');
const chart = LightweightCharts.createChart(cEl, {layout:{background:{type:'solid',color:'#0B0E14'},textColor:'#94A3B8'},grid:{vertLines:{color:'#161D2B'},horzLines:{color:'#161D2B'}},timeScale:{timeVisible:true}});
const cs = chart.addCandlestickSeries({upColor:'#10B981',downColor:'#F43F5E',borderVisible:false,wickUpColor:'#10B981',wickDownColor:'#F43F5E'});
new ResizeObserver(() => chart.applyOptions({width: cEl.clientWidth, height: cEl.clientHeight})).observe(cEl);
let pLines = [];
const ws = new WebSocket((location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws');
ws.onmessage = (e) => {
  const d = JSON.parse(e.data);
  if(d.init) cs.setData(d.init);
  if(d.candle) cs.update(d.candle);
  if(d.price) document.getElementById('price').textContent = '$' + d.price.toFixed(2);
  if(d.spread) document.getElementById('spread').textContent = 'Spread: ' + d.spread;
  const s = d.state;
  if(s){
    document.getElementById('bal').textContent = '$' + s.bal.toFixed(2);
    document.getElementById('eq').textContent = '$' + s.eq.toFixed(2);
    document.getElementById('wr').textContent = (s.tot ? ((s.win/s.tot)*100).toFixed(0) : 0) + '% (' + s.tot + ')';
    document.getElementById('hurst').textContent = s.h + ' (' + s.reg + ')';
    pLines.forEach(l => cs.removePriceLine(l)); pLines = [];
    if(s.pos){
      document.getElementById('pos-info').innerHTML = `<b class="${s.pos.side==='BUY'?'text-emerald-400':'text-rose-400'}">${s.pos.side}</b> @ $${s.pos.entry} | ربح: <b>$${s.pos.pnl}</b> ${s.pos.be?'(مؤمّنة)':''}`;
      pLines.push(cs.createPriceLine({price: s.pos.entry, color: '#38BDF8', title: 'ENTRY'}));
      pLines.push(cs.createPriceLine({price: s.pos.tp, color: '#10B981', title: 'TP'}));
      pLines.push(cs.createPriceLine({price: s.pos.sl, color: '#F43F5E', title: 'SL'}));
    } else { document.getElementById('pos-info').textContent = 'يراقب التدفق الخام...'; }
    document.getElementById('logs').innerHTML = s.logs.map(l => `<div class="p-1 bg-slate-900/60 rounded">${l}</div>`).join('');
  }
};
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
async def home(): return HTMLResponse(HTML)
  
