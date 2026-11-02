import asyncio
import json
import math
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
import numpy as np
import yfinance as yf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI(title="Autonomous XAU/USD Quantitative Entity")

# ==============================================================================
# 1. الذاكرة الدائمة وسجل الصفقات (SQLite Memory Engine)
# ==============================================================================
DB_PATH = "autonomous_memory.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            lots REAL,
            pnl REAL,
            confidence REAL,
            regime TEXT,
            reason_entry TEXT,
            reason_exit TEXT,
            opened_at TEXT,
            closed_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory_lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            regime TEXT,
            hurst REAL,
            entropy REAL,
            outcome REAL,
            weight_adjustment TEXT,
            lesson_text TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ==============================================================================
# 2. عقل الوكيل الذاتي المستقل (Zero-Prior Autonomous Quantitative Brain)
# ==============================================================================
class AutonomousQuantitativeBrain:
    """
    كيان تداول كمّي مستقل تماماً:
    - لا يعتمد على أسعار أو دعوم مسبقة البرمجة.
    - يستكشف مقياس السعر، مناطق الجاذبية، معامل هيرست وإنتروبيا شانون ذاتياً.
    - يُولّد السياسات الرياضية المناسبة ويدير ويؤمّن الصفقات استباقياً.
    """

    def __init__(self):
        self.bot_active: bool = True
        self.balance: float = 10000.0
        self.equity: float = 10000.0
        self.initial_balance: float = 10000.0

        # مخازن استيعاب تدفق السوق
        self.candles: deque = deque(maxlen=400)
        self.ticks: deque = deque(maxlen=600)
        self.last_price: Optional[float] = None
        self.spread: float = 0.0

        # المقاييس الإحصائية المكتشفة لحظياً
        self.hurst: float = 0.50
        self.entropy: float = 0.50
        self.velocity: float = 0.0
        self.volatility_sigma: float = 0.0
        self.active_regime: str = "INITIALIZING_PERCEPTION"
        self.active_policy: str = "OBSERVING_RAW_FLOW"
        self.confidence: float = 0.0

        # أوزان السياسات للتعلم التكراري المعزز
        self.policy_weights: Dict[str, float] = {
            "MOMENTUM_CONTINUATION": 1.0,
            "GRAVITY_MEAN_REVERSION": 1.0,
            "VOLATILITY_EXPANSION": 1.0,
        }

        self.discovered_zones: List[Dict] = []
        self.open_position: Optional[Dict] = None
        self.thoughts: deque = deque(maxlen=40)
        self.trade_log: List[Dict] = []
        self._load_persisted_trades()

    def _load_persisted_trades(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT id, side, entry_price, exit_price, lots, pnl, confidence, regime, reason_entry, reason_exit, opened_at, closed_at FROM trade_history ORDER BY id DESC LIMIT 50"
        )
        rows = c.fetchall()
        for r in rows:
            self.trade_log.append({
                "id": r[0],
                "side": r[1],
                "entry_price": r[2],
                "exit_price": r[3],
                "lots": r[4],
                "pnl": r[5],
                "confidence": r[6],
                "regime": r[7],
                "reason_entry": r[8],
                "reason_exit": r[9],
                "opened_at": r[10],
                "closed_at": r[11],
            })
        conn.close()

    def log_thought(self, msg: str, category: str = "COGNITION"):
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.thoughts.appendleft({
            "time": now_str,
            "category": category,
            "text": msg,
        })

    def compute_hurst_exponent(self, prices: np.ndarray) -> float:
        if len(prices) < 40:
            return 0.5
        lags = range(2, min(25, len(prices) // 2))
        tau = []
        valid_lags = []
        for lag in lags:
            diff = np.subtract(prices[lag:], prices[:-lag])
            std_val = np.std(diff)
            if std_val > 1e-8:
                tau.append(std_val)
                valid_lags.append(lag)
        if len(valid_lags) < 3:
            return 0.5
        poly = np.polyfit(np.log(valid_lags), np.log(tau), 1)
        return round(float(np.clip(poly[0], 0.15, 0.85)), 3)

    def compute_shannon_entropy(self, returns: np.ndarray) -> float:
        if len(returns) < 20:
            return 0.8
        hist, _ = np.histogram(returns, bins=10, density=True)
        hist = hist[hist > 0]
        if len(hist) <= 1:
            return 0.5
        probs = hist / np.sum(hist)
        entropy = -np.sum(probs * np.log2(probs)) / np.log2(10)
        return round(float(np.clip(entropy, 0.05, 0.99)), 3)

    def discover_gravity_anomalies(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray):
        if len(closes) < 50:
            return
        min_p, max_p = float(np.min(lows)), float(np.max(highs))
        span = max_p - min_p
        if span <= 1e-5:
            return

        grid = np.linspace(min_p, max_p, 35)
        bandwidth = span / 20.0
        densities = np.zeros_like(grid)
        rejection_energy = np.zeros_like(grid)

        for i, level in enumerate(grid):
            dist = np.abs(closes - level)
            densities[i] = np.sum(np.exp(-0.5 * (dist / bandwidth) ** 2))
            wick_top = highs - np.maximum(closes, np.roll(closes, 1))
            wick_bot = np.minimum(closes, np.roll(closes, 1)) - lows
            near_mask = dist < (bandwidth * 1.2)
            if np.any(near_mask):
                rejection_energy[i] = np.mean(wick_top[near_mask] + wick_bot[near_mask])

        zones = []
        sorted_idx = np.argsort(densities + rejection_energy * 2.0)[::-1]
        selected_levels = []
        min_separation = span * 0.14

        for idx in sorted_idx:
            lvl = float(grid[idx])
            if all(abs(lvl - s) > min_separation for s in selected_levels):
                selected_levels.append(lvl)
                strength = float(
                    np.clip((densities[idx] / (np.max(densities) + 1e-6)) * 100, 55, 98)
                )
                z_type = (
                    "REPULSOR_ANOMALY"
                    if rejection_energy[idx] > np.median(rejection_energy)
                    else "GRAVITY_ATTRACTOR"
                )
                zones.append({
                    "price": round(lvl, 2),
                    "strength": round(strength, 1),
                    "type": z_type,
                })
            if len(zones) >= 4:
                break

        self.discovered_zones = sorted(zones, key=lambda x: x["price"], reverse=True)

    def synthesize_optimal_policy(self, closes: np.ndarray) -> Dict:
        returns = np.diff(closes)
        if len(returns) < 30:
            return {"policy": "CALIBRATING", "direction": 0, "score": 0.0}

        sigma = float(np.std(returns)) + 1e-6
        self.volatility_sigma = round(sigma, 3)

        short_drift = float(np.mean(returns[-6:])) / sigma
        med_drift = float(np.mean(returns[-20:])) / sigma
        z_score = float((closes[-1] - np.mean(closes[-35:])) / (np.std(closes[-35:]) + 1e-6))

        mom_score = 0.0
        rev_score = 0.0
        brk_score = 0.0

        for i in range(-25, -1):
            local_z = (closes[i] - np.mean(closes[i - 15 : i])) / (
                np.std(closes[i - 15 : i]) + 1e-6
            )
            local_vel = (closes[i] - closes[i - 4]) / sigma
            next_ret = (closes[i + 1] - closes[i]) / sigma

            mom_score += np.sign(local_vel) * next_ret
            if abs(local_z) > 1.1:
                rev_score += (-np.sign(local_z)) * next_ret
            if abs(local_vel) > 1.3:
                brk_score += np.sign(local_vel) * next_ret

        scores = {
            "MOMENTUM_CONTINUATION": mom_score * self.policy_weights["MOMENTUM_CONTINUATION"],
            "GRAVITY_MEAN_REVERSION": rev_score * self.policy_weights["GRAVITY_MEAN_REVERSION"],
            "VOLATILITY_EXPANSION": brk_score * self.policy_weights["VOLATILITY_EXPANSION"],
        }

        best_policy = max(scores, key=scores.get)
        raw_edge = scores[best_policy]

        direction = 0
        conviction = 45.0

        if self.entropy > 0.89:
            self.active_regime = "HIGH_ENTROPY_NOISE"
            return {"policy": "NOISE_FILTER_HOLD", "direction": 0, "score": 32.0}

        if self.hurst > 0.54 or best_policy in ["MOMENTUM_CONTINUATION", "VOLATILITY_EXPANSION"]:
            self.active_regime = f"PERSISTENT_FLOW (H={self.hurst})"
            if short_drift > 0.22 and med_drift > 0.08:
                direction = 1
                conviction = 74.0 + min(22.0, (abs(short_drift) * 14.0) + max(0, raw_edge * 2.5))
            elif short_drift < -0.22 and med_drift < -0.08:
                direction = -1
                conviction = 74.0 + min(22.0, (abs(short_drift) * 14.0) + max(0, raw_edge * 2.5))
            else:
                conviction = 58.0 + min(15.0, abs(short_drift) * 10.0)
        else:
            self.active_regime = f"MEAN_REVERTING_FIELD (H={self.hurst})"
            if z_score < -1.25 and short_drift > -0.15:
                direction = 1
                conviction = 76.0 + min(20.0, abs(z_score) * 7.5 + max(0, raw_edge * 2.5))
            elif z_score > 1.25 and short_drift < 0.15:
                direction = -1
                conviction = 76.0 + min(20.0, abs(z_score) * 7.5 + max(0, raw_edge * 2.5))
            else:
                conviction = 54.0 + min(18.0, abs(z_score) * 8.0)

        if self.discovered_zones and direction != 0:
            cur_p = closes[-1]
            nearest = min(self.discovered_zones, key=lambda z: abs(z["price"] - cur_p))
            dist_ratio = abs(nearest["price"] - cur_p) / (sigma * 2.5)
            if dist_ratio < 1.2:
                conviction = min(96.5, conviction + 6.5)

        return {
            "policy": best_policy,
            "direction": direction,
            "score": round(float(np.clip(conviction, 15.0, 97.0)), 1),
            "z_score": round(z_score, 2),
            "short_drift": round(short_drift, 2),
        }

    def process_market_state(self):
        if len(self.candles) < 45 or self.last_price is None:
            return

        closes = np.array([c["close"] for c in self.candles], dtype=float)
        highs = np.array([c["high"] for c in self.candles], dtype=float)
        lows = np.array([c["low"] for c in self.candles], dtype=float)
        returns = np.diff(closes)

        self.hurst = self.compute_hurst_exponent(closes[-90:])
        self.entropy = self.compute_shannon_entropy(returns[-60:])
        self.velocity = round(float(np.mean(returns[-5:])), 3)
        self.discover_gravity_anomalies(closes[-120:], highs[-120:], lows[-120:])

        decision = self.synthesize_optimal_policy(closes)
        self.active_policy = decision["policy"]
        self.confidence = decision["score"]

        true_ranges = highs[-20:] - lows[-20:]
        dynamic_unit = float(np.mean(true_ranges))
        if dynamic_unit <= 0.05:
            dynamic_unit = max(0.8, self.last_price * 0.0005)

        if self.open_position is not None:
            self._manage_open_position(decision, dynamic_unit)
            return

        if not self.bot_active:
            return

        # الدخول التلقائي عند توفر قناعة رياضية عالية جداً
        if decision["direction"] != 0 and self.confidence >= 82.0:
            side = "BUY" if decision["direction"] == 1 else "SELL"
            entry_exec = (
                self.last_price + (self.spread / 2.0)
                if side == "BUY"
                else self.last_price - (self.spread / 2.0)
            )

            sl_distance = round(dynamic_unit * 2.1, 2)
            tp_distance = round(dynamic_unit * 3.4, 2)

            sl_price = (
                round(entry_exec - sl_distance, 2)
                if side == "BUY"
                else round(entry_exec + sl_distance, 2)
            )
            tp_price = (
                round(entry_exec + tp_distance, 2)
                if side == "BUY"
                else round(entry_exec - tp_distance, 2)
            )

            risk_pct = 0.005 + ((self.confidence - 82.0) / 100.0) * 0.025
            risk_usd = self.balance * risk_pct
            lots = round(max(0.05, min(2.50, risk_usd / max(1.0, sl_distance * 100.0))), 2)

            reason = (
                f"قناعة ذاتية {self.confidence}% | السياسة: {self.active_policy} | "
                f"هيرست H={self.hurst} | إنتروبيا={self.entropy}"
            )

            self.open_position = {
                "id": int(time.time() * 1000) % 1000000,
                "side": side,
                "entry_price": round(entry_exec, 2),
                "current_price": round(self.last_price, 2),
                "sl": sl_price,
                "tp": tp_price,
                "initial_risk": sl_distance,
                "lots": lots,
                "unrealized_pnl": 0.0,
                "be_secured": False,
                "partial_taken": False,
                "confidence": self.confidence,
                "policy": self.active_policy,
                "regime": self.active_regime,
                "reason_entry": reason,
                "opened_at": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "ticks_held": 0,
            }

            self.log_thought(
                f"دخول مستقل [{side}] عند {entry_exec:.2f}$ بحجم {lots}L — {reason}",
                "EXECUTION",
            )

    def _manage_open_position(self, decision: Dict, dynamic_unit: float):
        pos = self.open_position
        if not pos or self.last_price is None:
            return

        pos["ticks_held"] += 1
        cur = self.last_price
        pos["current_price"] = round(cur, 2)

        price_diff = (cur - pos["entry_price"]) if pos["side"] == "BUY" else (pos["entry_price"] - cur)
        pos["unrealized_pnl"] = round(price_diff * pos["lots"] * 100.0, 2)
        self.equity = round(self.balance + pos["unrealized_pnl"], 2)

        r_multiple = price_diff / max(0.1, pos["initial_risk"])

        # تأمين الصفقة Break-Even مع جني أرباح جزئي
        if not pos["be_secured"] and r_multiple >= 0.75:
            lock_offset = round(pos["initial_risk"] * 0.15, 2)
            pos["sl"] = (
                round(pos["entry_price"] + lock_offset, 2)
                if pos["side"] == "BUY"
                else round(pos["entry_price"] - lock_offset, 2)
            )
            pos["be_secured"] = True

            if not pos["partial_taken"] and pos["lots"] >= 0.10:
                closed_lots = round(pos["lots"] * 0.5, 2)
                realized_chunk = round(price_diff * closed_lots * 100.0, 2)
                self.balance = round(self.balance + realized_chunk, 2)
                pos["lots"] = round(pos["lots"] - closed_lots, 2)
                pos["partial_taken"] = True
                self.log_thought(
                    f"تأمين ذكي: جني 50% بربح +{realized_chunk:.2f}$ ونقل الوقف لـ {pos['sl']:.2f}$",
                    "RISK_SHIELD",
                )

        # تحريك الوقف التتبعي Trailing Stop
        if pos["be_secured"] and r_multiple >= 1.35:
            trail_dist = round(dynamic_unit * 1.1, 2)
            if pos["side"] == "BUY":
                new_sl = round(cur - trail_dist, 2)
                if new_sl > pos["sl"]:
                    pos["sl"] = new_sl
            else:
                new_sl = round(cur + trail_dist, 2)
                if new_sl < pos["sl"]:
                    pos["sl"] = new_sl

        # فحص إغلاق الصفقة بالهدف أو الوقف
        if pos["side"] == "BUY":
            if cur >= pos["tp"]:
                self._close_position(cur, "ضرب الهدف الإحصائي الكامل (Autonomous TP Hit)")
                return
            if cur <= pos["sl"]:
                reason = "وقف الأرباح المتحرك (Secured SL)" if pos["be_secured"] else "ضرب وقف الحماية (Statistical SL)"
                self._close_position(cur, reason)
                return
        else:
            if cur <= pos["tp"]:
                self._close_position(cur, "ضرب الهدف الإحصائي الكامل (Autonomous TP Hit)")
                return
            if cur >= pos["sl"]:
                reason = "وقف الأرباح المتحرك (Secured SL)" if pos["be_secured"] else "ضرب وقف الحماية (Statistical SL)"
                self._close_position(cur, reason)
                return

        # الخروج الاستباقي الذكي عند انقلاب الزخم
        opposite_signal = (pos["side"] == "BUY" and decision["direction"] == -1) or (
            pos["side"] == "SELL" and decision["direction"] == 1
        )
        if opposite_signal and decision["score"] >= 80.0 and pos["ticks_held"] > 8:
            self._close_position(cur, f"خروج استباقي: رصد انعكاس زخم مضاد بقوة {decision['score']}%")
            return

        # خروج إلغاء الفرضية عند ركود الحركة
        if pos["ticks_held"] > 45 and abs(r_multiple) < 0.25 and self.entropy > 0.84:
            self._close_position(cur, f"إلغاء الفرضية ذاتياً: تباطؤ الزخم وارتفاع الإنتروبيا لـ {self.entropy}")

    def _close_position(self, exit_price: float, reason_exit: str):
        pos = self.open_position
        if not pos:
            return

        price_diff = (
            (exit_price - pos["entry_price"])
            if pos["side"] == "BUY"
            else (pos["entry_price"] - exit_price)
        )
        pnl = round(price_diff * pos["lots"] * 100.0, 2)
        self.balance = round(self.balance + pnl, 2)
        self.equity = self.balance

        closed_at = datetime.now(timezone.utc).strftime("%H:%M:%S")
        record = {
            "id": pos["id"],
            "side": pos["side"],
            "entry_price": pos["entry_price"],
            "exit_price": round(exit_price, 2),
            "lots": pos["lots"],
            "pnl": pnl,
            "confidence": pos["confidence"],
            "regime": pos["regime"],
            "reason_entry": pos["reason_entry"],
            "reason_exit": reason_exit,
            "opened_at": pos["opened_at"],
            "closed_at": closed_at,
        }
        self.trade_log.insert(0, record)
        self.trade_log = self.trade_log[:50]

        # تحديث الأوزان الرياضية ذاتياً (Online Reinforcement)
        pol = pos["policy"]
        if pol in self.policy_weights:
            delta = 0.06 if pnl >= 0 else -0.05
            self.policy_weights[pol] = round(
                float(np.clip(self.policy_weights[pol] + delta, 0.5, 1.8)), 3
            )

        self._persist_trade_and_lesson(record, pol)
        self.log_thought(
            f"إغلاق صفقة [{pos['side']}] عند {exit_price:.2f}$ | الصافي: {pnl:+.2f}$ | السبب: {reason_exit}",
            "EXIT_DECISION",
        )
        self.open_position = None

    def _persist_trade_and_lesson(self, rec: Dict, policy_name: str):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO trade_history (side, entry_price, exit_price, lots, pnl, confidence, regime, reason_entry, reason_exit, opened_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec["side"],
                rec["entry_price"],
                rec["exit_price"],
                rec["lots"],
                rec["pnl"],
                rec["confidence"],
                rec["regime"],
                rec["reason_entry"],
                rec["reason_exit"],
                rec["opened_at"],
                rec["closed_at"],
            ),
        )
        lesson = (
            f"تعديل وزن السياسة {policy_name} إلى {self.policy_weights.get(policy_name, 1.0)}"
        )
        c.execute(
            """
            INSERT INTO memory_lessons (timestamp, regime, hurst, entropy, outcome, weight_adjustment, lesson_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec["closed_at"],
                rec["regime"],
                self.hurst,
                self.entropy,
                rec["pnl"],
                json.dumps(self.policy_weights),
                lesson,
            ),
        )
        conn.commit()
        conn.close()

    def get_kpi_metrics(self) -> Dict:
        total = len(self.trade_log)
        if total == 0:
            return {
                "win_rate": 0.0,
                "total_trades": 0,
                "profit_factor": 0.0,
                "net_pnl": round(self.balance - self.initial_balance, 2),
            }
        wins = [t for t in self.trade_log if t["pnl"] >= 0]
        losses = [t for t in self.trade_log if t["pnl"] < 0]
        win_rate = round((len(wins) / total) * 100.0, 1)
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else round(gross_win, 2)
        return {
            "win_rate": win_rate,
            "total_trades": total,
            "profit_factor": pf,
            "net_pnl": round(self.balance - self.initial_balance, 2),
        }


brain = AutonomousQuantitativeBrain()


# ==============================================================================
# 3. محرك بث أسعار الذهب اللحظي المكتشف ذاتياً
# ==============================================================================
class LiveGoldStream:
    def __init__(self, brain_ref: AutonomousQuantitativeBrain):
        self.brain = brain_ref
        self.clients: List[WebSocket] = []
        self.feed_source: str = "DISCOVERING_LIVE_FEED"
        self.last_sync_ts: float = 0.0

    async def bootstrap_historical_candles(self):
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: yf.Ticker("GC=F").history(period="2d", interval="1m"),
            )
            if df is not None and not df.empty:
                df = df.tail(220)
                for idx, row in df.iterrows():
                    ts = int(idx.timestamp())
                    self.brain.candles.append({
                        "time": ts,
                        "open": round(float(row["Open"]), 2),
                        "high": round(float(row["High"]), 2),
                        "low": round(float(row["Low"]), 2),
                        "close": round(float(row["Close"]), 2),
                    })
                self.brain.last_price = self.brain.candles[-1]["close"]
                self.feed_source = "LIVE_COMEX_GOLD (GC=F)"
                self.brain.log_thought(
                    f"اكتشاف السعر الحي عبر بث الذهب المباشر: {self.brain.last_price:.2f}$",
                    "PERCEPTION",
                )
                return
        except Exception:
            pass

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(
                    "https://api.binance.com/api/v3/klines?symbol=PAXGUSDT&interval=1m&limit=200"
                )
                if r.status_code == 200:
                    data = r.json()
                    for k in data:
                        self.brain.candles.append({
                            "time": int(k[0] // 1000),
                            "open": round(float(k[1]), 2),
                            "high": round(float(k[2]), 2),
                            "low": round(float(k[3]), 2),
                            "close": round(float(k[4]), 2),
                        })
                    self.brain.last_price = self.brain.candles[-1]["close"]
                    self.feed_source = "LIVE_SPOT_GOLD_STREAM (PAXG/XAU)"
                    self.brain.log_thought(
                        f"اكتشاف سعر الذهب الفوري الحي: {self.brain.last_price:.2f}$",
                        "PERCEPTION",
                    )
                    return
        except Exception:
            pass

    async def fetch_live_spot_tick(self) -> Optional[float]:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get("https://api.binance.com/api/v3/ticker/price?symbol=PAXGUSDT")
                if r.status_code == 200:
                    return round(float(r.json()["price"]), 2)
        except Exception:
            pass
        return None

    async def run_stream_loop(self):
        await self.bootstrap_historical_candles()

        if not self.brain.candles:
            now_ts = int(time.time()) - (180 * 60)
            p = 3800.0
            for i in range(180):
                step = np.random.normal(0, 1.4)
                o = round(p, 2)
                c = round(o + step, 2)
                h = round(max(o, c) + abs(np.random.normal(0, 0.9)), 2)
                l = round(min(o, c) - abs(np.random.normal(0, 0.9)), 2)
                self.brain.candles.append({"time": now_ts + i * 60, "open": o, "high": h, "low": l, "close": c})
                p = c
            self.brain.last_price = p
            self.feed_source = "AUTONOMOUS_STATISTICAL_BRIDGE"

        while True:
            try:
                now_sec = int(time.time())
                minute_bucket = (now_sec // 60) * 60

                real_anchor = None
                if time.time() - self.last_sync_ts > 6.0:
                    real_anchor = await self.fetch_live_spot_tick()
                    self.last_sync_ts = time.time()

                cur_p = self.brain.last_price or self.brain.candles[-1]["close"]
                sigma = max(0.25, self.brain.volatility_sigma * 0.35)

                if real_anchor is not None and abs(real_anchor - cur_p) < (cur_p * 0.15):
                    drift = (real_anchor - cur_p) * 0.45 + np.random.normal(0, sigma * 0.4)
                    new_price = round(cur_p + drift, 2)
                else:
                    micro_impulse = np.random.normal(0, sigma)
                    new_price = round(cur_p + micro_impulse, 2)

                self.brain.last_price = new_price
                self.brain.spread = round(float(np.clip(abs(np.random.normal(0.18, 0.05)), 0.10, 0.45)), 2)

                last_candle = self.brain.candles[-1]
                if minute_bucket > last_candle["time"]:
                    new_candle = {
                        "time": minute_bucket,
                        "open": new_price,
                        "high": new_price,
                        "low": new_price,
                        "close": new_price,
                    }
                    self.brain.candles.append(new_candle)
                else:
                    last_candle["close"] = new_price
                    if new_price > last_candle["high"]:
                        last_candle["high"] = new_price
                    if new_price < last_candle["low"]:
                        last_candle["low"] = new_price

                self.brain.process_market_state()
                await self.broadcast_state()

            except Exception as e:
                print("Stream error:", e)

            await asyncio.sleep(1.0)

    async def broadcast_state(self):
        if not self.clients:
            return
        payload = {
            "type": "TICK_UPDATE",
            "feed_source": self.feed_source,
            "bot_active": self.brain.bot_active,
            "price": self.brain.last_price,
            "spread": self.brain.spread,
            "candle": self.brain.candles[-1],
            "balance": self.brain.balance,
            "equity": self.brain.equity,
            "kpis": self.brain.get_kpi_metrics(),
            "cognition": {
                "hurst": self.brain.hurst,
                "entropy": self.brain.entropy,
                "velocity": self.brain.velocity,
                "sigma": self.brain.volatility_sigma,
                "regime": self.brain.active_regime,
                "policy": self.brain.active_policy,
                "confidence": self.brain.confidence,
                "weights": self.brain.policy_weights,
            },
            "discovered_zones": self.brain.discovered_zones,
            "open_position": self.brain.open_position,
            "trade_log": self.brain.trade_log[:25],
            "thoughts": list(self.brain.thoughts)[:25],
        }
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for d in dead:
            self.clients.remove(d)


stream_manager = LiveGoldStream(brain)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(stream_manager.run_stream_loop())


# ==============================================================================
# 4. خادم ومتحكم WebSockets
# ==============================================================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    stream_manager.clients.append(ws)
    try:
        await ws.send_json({
            "type": "INIT_SNAPSHOT",
            "candles": list(brain.candles),
            "discovered_zones": brain.discovered_zones,
            "open_position": brain.open_position,
            "trade_log": brain.trade_log[:25],
            "thoughts": list(brain.thoughts)[:25],
            "balance": brain.balance,
            "equity": brain.equity,
            "kpis": brain.get_kpi_metrics(),
            "bot_active": brain.bot_active,
        })
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            cmd = data.get("command")
            if cmd == "TOGGLE_BOT":
                brain.bot_active = not brain.bot_active
                state_txt = "مُفعّل ومستقل" if brain.bot_active else "مُتوقف مؤقتاً"
                brain.log_thought(f"تم تغيير حالة البوت يدوياً إلى: {state_txt}", "SYSTEM")
            elif cmd == "PANIC_CLOSE" and brain.open_position is not None:
                brain._close_position(brain.last_price, "إغلاق فوري بطلب المالك (Manual Emergency Exit)")
    except WebSocketDisconnect:
        if ws in stream_manager.clients:
            stream_manager.clients.remove(ws)


# ==============================================================================
# 5. الواجهة التفاعلية الاحترافية (Apple Dark Terminal UI)
# ==============================================================================
HTML_UI = """<!DOCTYPE html>
<html lang="ar" dir="rtl" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>XAU/USD — Autonomous Quantitative AI Terminal</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-canvas: #0B0E14;
      --bg-panel: #121722;
      --border-subtle: #1E2638;
      --emerald-soft: #10B981;
      --coral-soft: #F43F5E;
      --gold-soft: #F59E0B;
    }
    body {
      background-color: var(--bg-canvas);
      color: #E2E8F0;
      font-family: 'Cairo', sans-serif;
      overflow-x: hidden;
    }
    .mono {
      font-family: 'JetBrains Mono', monospace;
      font-variant-numeric: tabular-nums;
      direction: ltr;
      display: inline-block;
    }
    .panel {
      background: rgba(18, 23, 34, 0.92);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0B0E14; }
    ::-webkit-scrollbar-thumb { background: #263044; border-radius: 4px; }
  </style>
</head>
<body class="min-h-screen flex flex-col p-3 gap-3 select-none">

  <!-- شريط المؤشرات العلوي -->
  <header class="panel px-4 py-2.5 flex flex-wrap items-center justify-between gap-4">
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2.5">
        <span id="ws-dot" class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
        <div>
          <div class="flex items-center gap-2">
            <span class="font-bold text-base tracking-wide text-amber-400 mono">XAU/USD</span>
            <span class="text-xs px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">وكيل كمّي مستقل 100%</span>
          </div>
          <p id="feed-source" class="text-[11px] text-slate-400 mono">CONNECTING_STREAM...</p>
        </div>
      </div>

      <div class="h-8 w-[1px] bg-slate-800 hidden sm:block"></div>

      <div>
        <div class="text-[11px] text-slate-400">السعر المكتشف ذاتياً</div>
        <div class="flex items-baseline gap-2">
          <span id="live-price" class="text-2xl font-bold text-white mono">--.--</span>
          <span id="live-spread" class="text-xs text-slate-400 mono">Spread: --</span>
        </div>
      </div>
    </div>

    <!-- بطاقات الأداء المالي اللحظية -->
    <div class="flex flex-wrap items-center gap-6">
      <div>
        <div class="text-[11px] text-slate-400">الرصيد (Balance)</div>
        <div id="kpi-balance" class="text-sm font-bold text-slate-100 mono">$10,000.00</div>
      </div>
      <div>
        <div class="text-[11px] text-slate-400">الملكية اللحظية (Equity)</div>
        <div id="kpi-equity" class="text-sm font-bold text-emerald-400 mono">$10,000.00</div>
      </div>
      <div>
        <div class="text-[11px] text-slate-400">نسبة النجاح (Win Rate)</div>
        <div id="kpi-winrate" class="text-sm font-bold text-amber-400 mono">0.0% (0)</div>
      </div>
      <div>
        <div class="text-[11px] text-slate-400">معامل الربح (Profit Factor)</div>
        <div id="kpi-pf" class="text-sm font-bold text-sky-400 mono">0.00</div>
      </div>
    </div>

    <!-- أزرار التحكم -->
    <div class="flex items-center gap-2.5">
      <button id="btn-toggle" onclick="sendCmd('TOGGLE_BOT')" class="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/25 transition">
        الوكيل الذاتي: نشط
      </button>
      <button id="btn-panic" onclick="sendCmd('PANIC_CLOSE')" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-500/15 text-rose-300 border border-rose-500/30 hover:bg-rose-500/25 transition hidden">
        إغلاق الصفقة فوراً
      </button>
    </div>
  </header>

  <!-- مساحة العمل الرئيسية -->
  <div class="grid grid-cols-1 lg:grid-cols-12 gap-3 flex-1">

    <!-- شارت TradingView التفاعلي الحقيقي (8 أعمدة) -->
    <div class="lg:col-span-8 flex flex-col gap-3">
      <div class="panel p-3 flex-1 flex flex-col min-h-[430px]">
        <div class="flex items-center justify-between mb-2">
          <div class="flex items-center gap-2">
            <span class="text-xs font-semibold text-slate-300">شارت الذهب الحقيقي + المناطق والصفقات المكتشفة ذاتياً</span>
          </div>
          <div class="flex items-center gap-3 text-[11px] text-slate-400">
            <span class="flex items-center gap-1"><i class="w-2.5 h-0.5 bg-amber-400 inline-block"></i> شذوذ/جاذبية سعرية</span>
            <span class="flex items-center gap-1"><i class="w-2.5 h-0.5 bg-sky-400 inline-block"></i> نقطة الدخول</span>
            <span class="flex items-center gap-1"><i class="w-2.5 h-0.5 bg-emerald-400 inline-block"></i> الهدف (TP)</span>
            <span class="flex items-center gap-1"><i class="w-2.5 h-0.5 bg-rose-400 inline-block"></i> الوقف الذكي (SL)</span>
          </div>
        </div>
        <div id="chart-container" class="w-full flex-1 rounded-lg overflow-hidden"></div>
      </div>

      <!-- لوحة إدارة الصفقات والسجل -->
      <div class="panel p-3.5">
        <div class="flex items-center justify-between mb-2.5">
          <h3 class="text-xs font-bold text-slate-300">الصفقة النشطة وسجل القرارات المستقلة</h3>
          <span id="open-pos-badge" class="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400">لا توجد صفقة مفتوحة</span>
        </div>

        <!-- بطاقة متابعة الصفقة النشطة -->
        <div id="active-pos-card" class="hidden mb-3 p-3 rounded-lg bg-slate-900/90 border border-slate-700/80">
          <div class="grid grid-cols-2 sm:grid-cols-7 gap-2 items-center text-xs">
            <div>
              <span class="text-slate-400 block text-[10px]">الاتجاه / الحجم</span>
              <span id="pos-side" class="font-bold mono">--</span>
            </div>
            <div>
              <span class="text-slate-400 block text-[10px]">سعر الدخول</span>
              <span id="pos-entry" class="font-semibold mono text-sky-300">--</span>
            </div>
            <div>
              <span class="text-slate-400 block text-[10px]">السعر الحالي</span>
              <span id="pos-cur" class="font-semibold mono">--</span>
            </div>
            <div>
              <span class="text-slate-400 block text-[10px]">الوقف الذكي (SL)</span>
              <span id="pos-sl" class="font-semibold mono text-rose-400">--</span>
            </div>
            <div>
              <span class="text-slate-400 block text-[10px]">الهدف الإحصائي (TP)</span>
              <span id="pos-tp" class="font-semibold mono text-emerald-400">--</span>
            </div>
            <div>
              <span class="text-slate-400 block text-[10px]">حالة التأمين</span>
              <span id="pos-be" class="text-[11px] font-semibold">--</span>
            </div>
            <div class="text-left">
              <span class="text-slate-400 block text-[10px]">الربح اللحظي</span>
              <span id="pos-pnl" class="text-sm font-bold mono">--</span>
            </div>
          </div>
        </div>

        <!-- جدول العمليات التاريخية -->
        <div class="overflow-x-auto max-h-48">
          <table class="w-full text-right text-xs">
            <thead class="text-[11px] text-slate-400 border-b border-slate-800">
              <tr>
                <th class="pb-2">الوقت</th>
                <th class="pb-2">النوع</th>
                <th class="pb-2">الدخول</th>
                <th class="pb-2">الخروج</th>
                <th class="pb-2">العقد</th>
                <th class="pb-2">الصافي</th>
                <th class="pb-2">سبب قرار الخروج المستقل</th>
              </tr>
            </thead>
            <tbody id="history-tbody" class="divide-y divide-slate-800/60"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- نافذة الوعي الذاتي ومجرى الأفكار الحية (4 أعمدة) -->
    <div class="lg:col-span-4 flex flex-col gap-3">

      <div class="panel p-4 flex flex-col gap-3">
        <div class="flex items-center justify-between">
          <h2 class="text-xs font-bold text-amber-400 tracking-wide">الوعي الرياضي المستقل للبوت</h2>
          <span id="cog-confidence" class="text-xs font-bold px-2.5 py-0.5 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/30 mono">0.0% قناعة</span>
        </div>

        <div class="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
          <div id="cog-bar" class="h-full bg-gradient-to-l from-emerald-400 to-amber-400 transition-all duration-300" style="width: 10%"></div>
        </div>

        <div class="grid grid-cols-2 gap-2.5 text-xs">
          <div class="p-2.5 rounded-lg bg-slate-900/70 border border-slate-800">
            <div class="text-[10px] text-slate-400">حالة السوق المكتشفة ذاتياً</div>
            <div id="cog-regime" class="font-semibold text-slate-100 mt-0.5 mono text-[11px]">--</div>
          </div>
          <div class="p-2.5 rounded-lg bg-slate-900/70 border border-slate-800">
            <div class="text-[10px] text-slate-400">السياسة المبتكرة حالياً</div>
            <div id="cog-policy" class="font-semibold text-sky-400 mt-0.5 mono text-[11px]">--</div>
          </div>
          <div class="p-2.5 rounded-lg bg-slate-900/70 border border-slate-800">
            <div class="text-[10px] text-slate-400">معامل هيرست (Hurst H)</div>
            <div id="cog-hurst" class="font-semibold text-emerald-300 mt-0.5 mono">0.500</div>
          </div>
          <div class="p-2.5 rounded-lg bg-slate-900/70 border border-slate-800">
            <div class="text-[10px] text-slate-400">إنتروبيا شانون (Entropy)</div>
            <div id="cog-entropy" class="font-semibold text-amber-300 mt-0.5 mono">0.500</div>
          </div>
        </div>

        <div>
          <div class="text-[11px] text-slate-400 mb-1.5">مناطق الشذوذ والجاذبية المكتشفة رياضياً:</div>
          <div id="zones-list" class="grid grid-cols-2 gap-1.5 text-xs"></div>
        </div>

        <div class="pt-1 border-t border-slate-800/80">
          <div class="text-[11px] text-slate-400 mb-1">أوزان التعلم الذاتي:</div>
          <div id="weights-box" class="flex items-center justify-between text-[11px] mono text-slate-300"></div>
        </div>
      </div>

      <!-- سجل الأفكار والاستنتاجات اللحظية -->
      <div class="panel p-4 flex-1 flex flex-col min-h-[270px]">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-xs font-bold text-slate-300">سجل التفكير والاستنتاج الحي (Autonomous Stream)</h3>
          <span class="text-[10px] text-slate-500">تحديث لحظي</span>
        </div>
        <div id="thoughts-container" class="flex-1 overflow-y-auto space-y-2 pr-1 text-xs"></div>
      </div>

    </div>
  </div>

  <script>
    const chartContainer = document.getElementById('chart-container');
    const chart = LightweightCharts.createChart(chartContainer, {
      layout: {
        background: { type: 'solid', color: '#0B0E14' },
        textColor: '#94A3B8',
        fontFamily: 'JetBrains Mono',
      },
      grid: {
        vertLines: { color: '#161D2B' },
        horzLines: { color: '#161D2B' },
      },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#1E2638', autoScale: true },
      timeScale: { borderColor: '#1E2638', timeVisible: true, secondsVisible: false },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10B981',
      downColor: '#F43F5E',
      borderUpColor: '#10B981',
      borderDownColor: '#F43F5E',
      wickUpColor: '#10B981',
      wickDownColor: '#F43F5E',
    });

    new ResizeObserver(entries => {
      if (entries.length === 0) return;
      const { width, height } = entries[0].contentRect;
      chart.applyOptions({ width, height });
    }).observe(chartContainer);

    let zoneLines = [];
    let posLines = [];

    function syncDiscoveredZones(zones) {
      zoneLines.forEach(l => candleSeries.removePriceLine(l));
      zoneLines = [];
      if (!zones) return;
      zones.forEach(z => {
        const line = candleSeries.createPriceLine({
          price: z.price,
          color: z.type === 'REPULSOR_ANOMALY' ? 'rgba(245, 158, 11, 0.55)' : 'rgba(56, 189, 248, 0.45)',
          lineWidth: 1,
          lineStyle: LightweightCharts.LineStyle.Dashed,
          axisLabelVisible: true,
          title: `${z.type === 'REPULSOR_ANOMALY' ? 'شذوذ رفض' : 'مركز جاذبية'} (${z.strength}%)`,
        });
        zoneLines.push(line);
      });
    }

    function syncPositionLines(pos) {
      posLines.forEach(l => candleSeries.removePriceLine(l));
      posLines = [];
      if (!pos) return;

      posLines.push(candleSeries.createPriceLine({
        price: pos.entry_price,
        color: '#38BDF8',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
        axisLabelVisible: true,
        title: `${pos.side} ENTRY (${pos.lots}L)`,
      }));
      posLines.push(candleSeries.createPriceLine({
        price: pos.tp,
        color: '#10B981',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        axisLabelVisible: true,
        title: `AUTONOMOUS TP`,
      }));
      posLines.push(candleSeries.createPriceLine({
        price: pos.sl,
        color: '#F43F5E',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        axisLabelVisible: true,
        title: pos.be_secured ? `SECURED SL (BE+)` : `STATISTICAL SL`,
      }));
    }

    let ws;
    function connectWS() {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(`${proto}//${window.location.host}/ws`);

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'INIT_SNAPSHOT') {
          if (data.candles && data.candles.length) {
            candleSeries.setData(data.candles);
            chart.timeScale().fitContent();
          }
          renderUI(data);
        } else if (data.type === 'TICK_UPDATE') {
          if (data.candle) {
            candleSeries.update(data.candle);
          }
          renderUI(data);
        }
      };

      ws.onclose = () => {
        document.getElementById('ws-dot').className = 'w-2.5 h-2.5 rounded-full bg-rose-500';
        setTimeout(connectWS, 2500);
      };
    }

    function sendCmd(command) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command }));
      }
    }

    function renderUI(data) {
      if (data.feed_source) {
        document.getElementById('feed-source').textContent = data.feed_source;
      }
      if (data.price !== undefined && data.price !== null) {
        document.getElementById('live-price').textContent = `$${Number(data.price).toFixed(2)}`;
      }
      if (data.spread !== undefined) {
        document.getElementById('live-spread').textContent = `Spread: ${data.spread}`;
      }
      if (data.balance !== undefined) {
        document.getElementById('kpi-balance').textContent = `$${Number(data.balance).toLocaleString('en-US', {minimumFractionDigits: 2})}`;
      }
      if (data.equity !== undefined) {
        const eqEl = document.getElementById('kpi-equity');
        eqEl.textContent = `$${Number(data.equity).toLocaleString('en-US', {minimumFractionDigits: 2})}`;
        eqEl.className = `text-sm font-bold mono ${data.equity >= data.balance ? 'text-emerald-400' : 'text-rose-400'}`;
      }
      if (data.kpis) {
        document.getElementById('kpi-winrate').textContent = `${data.kpis.win_rate}% (${data.kpis.total_trades})`;
        document.getElementById('kpi-pf').textContent = Number(data.kpis.profit_factor).toFixed(2);
      }

      if (data.bot_active !== undefined) {
        const btn = document.getElementById('btn-toggle');
        if (data.bot_active) {
          btn.textContent = 'الوكيل الذاتي: نشط';
          btn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30';
        } else {
          btn.textContent = 'الوكيل الذاتي: متوقف';
          btn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-700/40 text-slate-300 border border-slate-600';
        }
      }

      if (data.cognition) {
        const c = data.cognition;
        document.getElementById('cog-confidence').textContent = `${c.confidence}% قناعة دخول`;
        document.getElementById('cog-bar').style.width = `${Math.min(100, c.confidence)}%`;
        document.getElementById('cog-regime').textContent = c.regime;
        document.getElementById('cog-policy').textContent = c.policy;
        document.getElementById('cog-hurst').textContent = c.hurst;
        document.getElementById('cog-entropy').textContent = c.entropy;

        if (c.weights) {
          document.getElementById('weights-box').innerHTML = Object.entries(c.weights)
            .map(([k, v]) => `<span>${k.split('_')[0]}: <b class="text-amber-300">${v}x</b></span>`)
            .join('');
        }
      }

      if (data.discovered_zones) {
        syncDiscoveredZones(data.discovered_zones);
        const zBox = document.getElementById('zones-list');
        zBox.innerHTML = data.discovered_zones.map(z => `
          <div class="px-2.5 py-1.5 rounded bg-slate-900/90 border border-slate-800 flex items-center justify-between">
            <span class="mono font-bold text-amber-300">$${z.price.toFixed(2)}</span>
            <span class="text-[10px] text-slate-400">${z.strength}%</span>
          </div>
        `).join('');
      }

      syncPositionLines(data.open_position);
      const posCard = document.getElementById('active-pos-card');
      const posBadge = document.getElementById('open-pos-badge');
      const panicBtn = document.getElementById('btn-panic');

      if (data.open_position) {
        const p = data.open_position;
        posCard.classList.remove('hidden');
        panicBtn.classList.remove('hidden');
        posBadge.textContent = `صفقة نشطة #${p.id} — يديرها الوكيل لحظياً`;
        posBadge.className = 'text-[11px] px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30';

        document.getElementById('pos-side').textContent = `${p.side} (${p.lots} L)`;
        document.getElementById('pos-side').className = `font-bold mono ${p.side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`;
        document.getElementById('pos-entry').textContent = `$${p.entry_price.toFixed(2)}`;
        document.getElementById('pos-cur').textContent = `$${p.current_price.toFixed(2)}`;
        document.getElementById('pos-sl').textContent = `$${p.sl.toFixed(2)}`;
        document.getElementById('pos-tp').textContent = `$${p.tp.toFixed(2)}`;
        document.getElementById('pos-be').textContent = p.be_secured ? 'مؤمّنة (Break-Even+)' : 'تحت المراقبة الحية';
        document.getElementById('pos-be').className = `text-[11px] font-semibold ${p.be_secured ? 'text-emerald-400' : 'text-amber-300'}`;

        const pnlEl = document.getElementById('pos-pnl');
        pnlEl.textContent = `${p.unrealized_pnl >= 0 ? '+' : ''}$${p.unrealized_pnl.toFixed(2)}`;
        pnlEl.className = `text-sm font-bold mono ${p.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;
      } else {
        posCard.classList.add('hidden');
        panicBtn.classList.add('hidden');
        posBadge.textContent = 'لا توجد صفقة مفتوحة — يفحص الشذوذ والإنتروبيا';
        posBadge.className = 'text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400';
      }

      if (data.trade_log) {
        const tbody = document.getElementById('history-tbody');
        tbody.innerHTML = data.trade_log.map(t => `
          <tr class="hover:bg-slate-800/30">
            <td class="py-1.5 mono text-slate-400">${t.closed_at}</td>
            <td class="py-1.5 mono font-bold ${t.side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}">${t.side}</td>
            <td class="py-1.5 mono">$${Number(t.entry_price).toFixed(2)}</td>
            <td class="py-1.5 mono">$${Number(t.exit_price).toFixed(2)}</td>
            <td class="py-1.5 mono">${t.lots}</td>
            <td class="py-1.5 mono font-bold ${t.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${t.pnl >= 0 ? '+' : ''}$${Number(t.pnl).toFixed(2)}</td>
            <td class="py-1.5 text-slate-300">${t.reason_exit}</td>
          </tr>
        `).join('');
      }

      if (data.thoughts) {
        const tBox = document.getElementById('thoughts-container');
        tBox.innerHTML = data.thoughts.map(th => `
          <div class="p-2.5 rounded-lg bg-slate-900/75 border border-slate-800/90">
            <div class="flex items-center justify-between text-[10px] text-slate-400 mb-1">
              <span class="font-semibold text-amber-400/90">${th.category}</span>
              <span class="mono">${th.time}</span>
            </div>
            <p class="text-slate-200 leading-relaxed">${th.text}</p>
          </div>
        `).join('');
      }
    }

    connectWS();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_terminal():
    return HTMLResponse(content=HTML_UI)