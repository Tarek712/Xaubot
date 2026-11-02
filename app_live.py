import asyncio
import json
import os
import sqlite3
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
import numpy as np
import yfinance as yf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ==============================================================================
# 0. إعدادات قابلة للتعديل
# ==============================================================================
DB_PATH = os.getenv("BOT_DB_PATH", "autonomous_memory.db")

# --- مصدر السعر: CFD الذهب الفوري XAU/USD من Swissquote (Bid/Ask حقيقي، بلا مفتاح) ---
SQ_URL = "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD"
SQ_PROFILE = os.getenv("SQ_PROFILE", "standard").lower()  # standard | premium | prime ...
POLL_S = 2.0                     # تكرار جلب السعر (ثوانٍ)
STALE_QUOTE_S = 120              # أكبر عمر مقبول للسعر قبل اعتبار السوق مغلقاً
TF_MIN = int(os.getenv("BOT_TF_MIN", "1"))  # الإطار الزمني بالدقائق: 1 | 5 | 15
_YF = {1: "1m", 5: "5m", 15: "15m"}
if TF_MIN not in _YF:
    raise SystemExit("BOT_TF_MIN must be one of 1, 5, 15")
YF_INTERVAL = _YF[TF_MIN]
TF_SEC = TF_MIN * 60
MIN_CANDLES = 46                 # أقل عدد شموع قبل بدء التحليل
MAX_SPREAD = 2.0                 # لا دخول إذا اتسع السبريد فوق هذا ($)
ASSUMED_SPREAD = 0.25            # سبريد ثابت افتراضي ($) للتداول الورقي
ENTRY_CONFIDENCE = 82.0          # الحد الأدنى للقناعة للدخول
SIGNAL_CONFIRM = 2               # عدد الشموع المغلقة المتتالية لتأكيد الإشارة قبل الدخول
COOLDOWN_AFTER_WIN_S = 120 * TF_MIN  # انتظار بعد صفقة رابحة
COOLDOWN_AFTER_LOSS_S = 300 * TF_MIN  # انتظار بعد صفقة خاسرة
MAX_CONSECUTIVE_LOSSES = 4       # عدد الخسائر المتتالية قبل الإيقاف المؤقت
LOSS_STREAK_PAUSE_S = 1800 * TF_MIN  # الإيقاف بعد سلسلة الخسائر
OPPOSITE_EXIT_MIN_HOLD_S = 180 * TF_MIN  # أقل مدة احتفاظ قبل الخروج بإشارة معاكسة
STAGNATION_EXIT_S = 900 * TF_MIN  # مدة الركود قبل إلغاء الفرضية


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%m-%d %H:%M:%S")


# ==============================================================================
# 1. الذاكرة الدائمة وسجل الصفقات (SQLite Memory Engine)
# ==============================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
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
    finally:
        conn.close()


init_db()


# ==============================================================================
# 2. عقل الوكيل الذاتي المستقل
# ==============================================================================
class AutonomousQuantitativeBrain:
    POLICIES = (
        "MOMENTUM_CONTINUATION",
        "GRAVITY_MEAN_REVERSION",
        "VOLATILITY_EXPANSION",
    )

    def __init__(self):
        self.bot_active: bool = True
        self.can_trade: bool = False  # يحدده محرك البث (بث حقيقي وحديث فقط)
        self.balance: float = 10000.0
        self.equity: float = 10000.0
        self.initial_balance: float = 10000.0

        self.candles: deque = deque(maxlen=400)
        self.last_price: Optional[float] = None
        self.spread: float = ASSUMED_SPREAD

        self.hurst: float = 0.50
        self.entropy: float = 0.50
        self.velocity: float = 0.0
        self.volatility_sigma: float = 0.0
        self.dynamic_unit: float = 1.0
        self.hist_hurst: deque = deque(maxlen=300)
        self.hist_entropy: deque = deque(maxlen=300)
        self.hurst_rank: float = 0.5
        self.entropy_rank: float = 0.5
        self.block_reason: str = "تجميع البيانات..."
        self._streak: int = 0
        self._last_dir: int = 0
        self.active_regime: str = "INITIALIZING_PERCEPTION"
        self.active_policy: str = "OBSERVING_RAW_FLOW"
        self.confidence: float = 0.0

        self.policy_weights: Dict[str, float] = {p: 1.0 for p in self.POLICIES}

        self.discovered_zones: List[Dict] = []
        self.open_position: Optional[Dict] = None
        self.thoughts: deque = deque(maxlen=40)
        self.trade_log: List[Dict] = []
        self.stats = {"total": 0, "wins": 0, "gross_win": 0.0, "gross_loss": 0.0}
        self.consec_losses: int = 0
        self.cooldown_until: float = 0.0

        self._analysis_ts: Optional[int] = None
        self._decision: Dict = {"policy": "CALIBRATING", "direction": 0, "score": 0.0}
        self._restore_state()

    # ----------------------------------------------------------------- الذاكرة
    def _restore_state(self):
        """استرجاع الرصيد والأوزان والإحصائيات من قاعدة البيانات بعد إعادة التشغيل."""
        try:
            conn = sqlite3.connect(DB_PATH)
            try:
                c = conn.cursor()
                c.execute(
                    "SELECT id, side, entry_price, exit_price, lots, pnl, confidence, regime, "
                    "reason_entry, reason_exit, opened_at, closed_at "
                    "FROM trade_history ORDER BY id DESC LIMIT 50"
                )
                for r in c.fetchall():
                    self.trade_log.append({
                        "id": r[0], "side": r[1], "entry_price": r[2], "exit_price": r[3],
                        "lots": r[4], "pnl": r[5], "confidence": r[6], "regime": r[7],
                        "reason_entry": r[8], "reason_exit": r[9],
                        "opened_at": r[10], "closed_at": r[11],
                    })

                c.execute(
                    "SELECT COUNT(*), "
                    "COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0), "
                    "COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0), "
                    "COALESCE(SUM(CASE WHEN pnl < 0 THEN -pnl ELSE 0 END), 0), "
                    "COALESCE(SUM(pnl), 0) FROM trade_history"
                )
                total, wins, gw, gl, net = c.fetchone()
                self.stats = {
                    "total": int(total), "wins": int(wins),
                    "gross_win": float(gw), "gross_loss": float(gl),
                }
                self.balance = round(self.initial_balance + float(net), 2)
                self.equity = self.balance

                c.execute("SELECT weight_adjustment FROM memory_lessons ORDER BY id DESC LIMIT 1")
                row = c.fetchone()
                if row and row[0]:
                    saved = json.loads(row[0])
                    for k in self.policy_weights:
                        if k in saved:
                            self.policy_weights[k] = float(np.clip(float(saved[k]), 0.5, 1.8))
            finally:
                conn.close()
        except Exception as e:
            print("Restore error:", repr(e))

        streak = 0
        for t in self.trade_log:  # الأحدث أولاً
            if t["pnl"] < 0:
                streak += 1
            else:
                break
        self.consec_losses = streak

    def log_thought(self, msg: str, category: str = "COGNITION"):
        self.thoughts.appendleft({
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "category": category,
            "text": msg,
        })

    # --------------------------------------------------------------- الإحصاء
    def compute_hurst_exponent(self, prices: np.ndarray) -> float:
        if len(prices) < 40:
            return 0.5
        lags = range(2, min(25, len(prices) // 2))
        tau, valid_lags = [], []
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

    def discover_gravity_anomalies(
        self, opens: np.ndarray, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray
    ):
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

        # فتائل الشموع الحقيقية (باستخدام الافتتاح بدل الإغلاق السابق، وبدون التفاف np.roll)
        wicks = (highs - np.maximum(opens, closes)) + (np.minimum(opens, closes) - lows)

        for i, level in enumerate(grid):
            dist = np.abs(closes - level)
            densities[i] = np.sum(np.exp(-0.5 * (dist / bandwidth) ** 2))
            near_mask = dist < (bandwidth * 1.2)
            if np.any(near_mask):
                rejection_energy[i] = float(np.mean(wicks[near_mask]))

        # توحيد المقاييس قبل الدمج (الكثافة بالعدد والفتائل بالدولار)
        d_norm = densities / (np.max(densities) + 1e-9)
        r_norm = rejection_energy / (np.max(rejection_energy) + 1e-9)
        combined = d_norm + 0.6 * r_norm
        cmax = float(np.max(combined)) + 1e-9
        med_rej = float(np.median(rejection_energy))

        zones, selected_levels = [], []
        min_separation = span * 0.14
        for idx in np.argsort(combined)[::-1]:
            lvl = float(grid[idx])
            if all(abs(lvl - s) > min_separation for s in selected_levels):
                selected_levels.append(lvl)
                zones.append({
                    "price": round(lvl, 2),
                    "strength": round(min(98.0, float(combined[idx]) / cmax * 100.0), 1),
                    "type": "REPULSOR_ANOMALY" if rejection_energy[idx] > med_rej else "GRAVITY_ATTRACTOR",
                })
            if len(zones) >= 4:
                break

        self.discovered_zones = sorted(zones, key=lambda x: x["price"], reverse=True)

    def synthesize_optimal_policy(self, closes: np.ndarray) -> Dict:
        returns = np.diff(closes)
        if len(returns) < 30:
            self.active_regime = "CALIBRATING"
            return {"policy": "CALIBRATING", "direction": 0, "score": 0.0, "reason": "معايرة: بيانات غير كافية"}

        sigma = float(np.std(returns)) + 1e-6
        self.volatility_sigma = round(sigma, 3)

        short_drift = float(np.mean(returns[-6:])) / sigma
        med_drift = float(np.mean(returns[-20:])) / sigma
        z_score = float((closes[-1] - np.mean(closes[-35:])) / (np.std(closes[-35:]) + 1e-6))

        mom_score = rev_score = brk_score = 0.0
        for i in range(-25, -1):
            local_z = (closes[i] - np.mean(closes[i - 15: i])) / (np.std(closes[i - 15: i]) + 1e-6)
            local_vel = (closes[i] - closes[i - 4]) / sigma
            next_ret = (closes[i + 1] - closes[i]) / sigma
            mom_score += np.sign(local_vel) * next_ret
            if abs(local_z) > 1.1:
                rev_score += (-np.sign(local_z)) * next_ret
            if abs(local_vel) > 1.3:
                brk_score += np.sign(local_vel) * next_ret

        # الوزن يُضخّم الميزة الموجبة ويُخفّف السالبة (الضرب المباشر كان يعاقب السياسة ذات الوزن الأعلى)
        def weighted(policy: str, s: float) -> float:
            w = self.policy_weights[policy]
            return s * w if s >= 0 else s / w

        scores = {
            "MOMENTUM_CONTINUATION": weighted("MOMENTUM_CONTINUATION", mom_score),
            "GRAVITY_MEAN_REVERSION": weighted("GRAVITY_MEAN_REVERSION", rev_score),
            "VOLATILITY_EXPANSION": weighted("VOLATILITY_EXPANSION", brk_score),
        }

        # عتبات نسبية لسجل البوت نفسه (المطلقة 0.89 / 0.54 / 0.46 كانت تحجب السوق معظم الوقت)
        if len(self.hist_entropy) >= 60 and self.entropy_rank > 0.90:
            self.active_regime = "HIGH_ENTROPY_NOISE"
            return {"policy": "NOISE_FILTER_HOLD", "direction": 0, "score": 32.0,
                    "reason": f"ضجيج مرتفع (إنتروبيا {self.entropy} ضمن أعلى 10% حديثاً)"}

        # اختيار السياسة من نظام السوق (هيرست) — والاسم المعروض هو السياسة المنفَّذة فعلاً
        if self.hurst_rank > 0.65:
            candidates = ("MOMENTUM_CONTINUATION", "VOLATILITY_EXPANSION")
            regime_name = "PERSISTENT_FLOW"
        elif self.hurst_rank < 0.35:
            candidates = ("GRAVITY_MEAN_REVERSION",)
            regime_name = "MEAN_REVERTING_FIELD"
        else:
            candidates = tuple(scores.keys())
            regime_name = "NEUTRAL_FIELD"
        policy = max(candidates, key=lambda p: scores[p])
        raw_edge = scores[policy]
        trend_mode = policy != "GRAVITY_MEAN_REVERSION"
        self.active_regime = f"{regime_name} (H={self.hurst})"

        direction = 0
        conviction = 45.0
        reason = ""
        if trend_mode:
            if short_drift > 0.22 and med_drift > 0.08:
                direction = 1
            elif short_drift < -0.22 and med_drift < -0.08:
                direction = -1
            if direction != 0:
                conviction = 74.0 + min(22.0, abs(short_drift) * 14.0 + max(0.0, raw_edge * 2.5))
            else:
                conviction = 58.0 + min(15.0, abs(short_drift) * 10.0)
        else:
            if z_score < -1.25 and short_drift > -0.15:
                direction = 1
            elif z_score > 1.25 and short_drift < 0.15:
                direction = -1
            if direction != 0:
                conviction = 76.0 + min(20.0, abs(z_score) * 7.5 + max(0.0, raw_edge * 2.5))
            else:
                conviction = 54.0 + min(18.0, abs(z_score) * 8.0)

        # لا دخول إذا لم تُظهر السياسة ميزة إحصائية حديثة
        if direction != 0 and raw_edge <= 0:
            direction = 0
            conviction = min(conviction, 70.0)
            reason = "السياسة بلا ميزة إحصائية حديثة"

        # المناطق: دعم/مقاومة خلف الصفقة تعزز القناعة، ومنطقة أمامها تعيق الهدف فتخفضها
        if self.discovered_zones and direction != 0:
            cur_p = self.last_price if self.last_price is not None else float(closes[-1])
            nearest = min(self.discovered_zones, key=lambda z: abs(z["price"] - cur_p))
            gap = nearest["price"] - cur_p
            if abs(gap) < sigma * 3.0:
                behind = (gap <= 0) if direction == 1 else (gap >= 0)
                conviction += 6.5 if behind else -8.0

        final = round(float(np.clip(conviction, 15.0, 96.5)), 1)
        if direction == 0 and not reason:
            reason = "لا إعداد: لا زخم كافٍ" if trend_mode else "لا إعداد: السعر قريب من متوسطه"
        elif direction != 0 and final < ENTRY_CONFIDENCE:
            reason = f"إعداد موجود لكن القناعة {final}% < {ENTRY_CONFIDENCE}%"
        elif direction != 0:
            reason = "إشارة جاهزة"
        return {
            "policy": policy,
            "direction": direction,
            "score": final,
            "reason": reason,
            "z_score": round(z_score, 2),
            "short_drift": round(short_drift, 2),
        }

    # ----------------------------------------------------------- دورة القرار
    @staticmethod
    def _rank(history, x: float) -> float:
        arr = np.asarray(history, dtype=float)
        return float(np.mean(arr <= x)) if len(arr) else 0.5

    def _warm_history(self, closes: np.ndarray):
        """تعبئة سجل هيرست/الإنتروبيا من التاريخ لتعمل العتبات النسبية فوراً."""
        start = max(91, len(closes) - 300)
        for j in range(start, len(closes)):
            self.hist_hurst.append(self.compute_hurst_exponent(closes[j - 90:j]))
            self.hist_entropy.append(self.compute_shannon_entropy(np.diff(closes[max(0, j - 61):j])))

    def _analyze(self, closed: List[Dict]):
        """تحليل ثقيل على الشموع المغلقة فقط — يُنفَّذ مرة واحدة عند كل شمعة جديدة."""
        opens = np.array([c["open"] for c in closed], dtype=float)
        closes = np.array([c["close"] for c in closed], dtype=float)
        highs = np.array([c["high"] for c in closed], dtype=float)
        lows = np.array([c["low"] for c in closed], dtype=float)
        returns = np.diff(closes)

        if len(self.hist_hurst) < 60 and len(closes) >= 160:
            self._warm_history(closes)
        self.hurst = self.compute_hurst_exponent(closes[-90:])
        self.entropy = self.compute_shannon_entropy(returns[-60:])
        self.hist_hurst.append(self.hurst)
        self.hist_entropy.append(self.entropy)
        self.hurst_rank = self._rank(self.hist_hurst, self.hurst)
        self.entropy_rank = self._rank(self.hist_entropy, self.entropy)
        self.velocity = round(float(np.mean(returns[-5:])), 3)
        self.discover_gravity_anomalies(opens[-120:], closes[-120:], highs[-120:], lows[-120:])

        self._decision = self.synthesize_optimal_policy(closes)
        self.active_policy = self._decision["policy"]
        self.confidence = self._decision["score"]

        # تأكيد الإشارة: نفس الاتجاه بقناعة كافية على شمعتين مغلقتين متتاليتين (يقلل الدخول على ضجيج لحظي)
        d = self._decision
        if d["direction"] != 0 and d["score"] >= ENTRY_CONFIDENCE:
            self._streak = self._streak + 1 if d["direction"] == self._last_dir else 1
            self._last_dir = d["direction"]
        else:
            self._streak, self._last_dir = 0, 0
        d["confirmed"] = self._streak >= SIGNAL_CONFIRM
        if d["direction"] != 0 and d["score"] >= ENTRY_CONFIDENCE and not d["confirmed"]:
            d["reason"] = f"إشارة أولية — بانتظار التأكيد ({self._streak}/{SIGNAL_CONFIRM} شموع)"

        unit = float(np.mean(highs[-20:] - lows[-20:]))
        if unit <= 0.05:
            unit = max(0.8, (self.last_price or float(closes[-1])) * 0.0005)
        self.dynamic_unit = unit

    def process_market_state(self):
        if len(self.candles) < MIN_CANDLES or self.last_price is None:
            self.block_reason = f"تجميع البيانات ({len(self.candles)}/{MIN_CANDLES} شمعة)"
            return

        forming_ts = self.candles[-1]["time"]
        if forming_ts != self._analysis_ts:
            self._analysis_ts = forming_ts
            self._analyze(list(self.candles)[:-1])  # استبعاد الشمعة الجارية

        decision = self._decision

        if self.open_position is not None:
            self.block_reason = "صفقة مفتوحة قيد الإدارة"
            self._manage_open_position(decision)
            return

        self.equity = self.balance
        if not self.bot_active:
            self.block_reason = "البوت متوقف يدوياً"
            return
        if not self.can_trade:
            self.block_reason = "البث غير حي (السوق مغلق أو لا أسعار)"
            return
        remaining = self.cooldown_until - time.time()
        if remaining > 0:
            self.block_reason = f"فترة تهدئة بعد آخر صفقة: {int(remaining)}ث"
            return
        if self.spread > MAX_SPREAD:
            self.block_reason = f"السبريد واسع ({self.spread}$ > {MAX_SPREAD}$)"
            return
        if decision["direction"] != 0 and decision["score"] >= ENTRY_CONFIDENCE and decision.get("confirmed"):
            self.block_reason = "إشارة مؤكدة — تنفيذ"
            self._open_position(decision)
        else:
            self.block_reason = decision.get("reason", "لا إعداد")

    def _exit_quote(self, side: str) -> float:
        """سعر الإغلاق الفعلي: البيع على Bid للشراء، والشراء على Ask للبيع."""
        half = self.spread / 2.0
        p = self.last_price - half if side == "BUY" else self.last_price + half
        return round(p, 2)

    def _open_position(self, decision: Dict):
        if self.balance <= 0 or self.last_price is None:
            return
        side = "BUY" if decision["direction"] == 1 else "SELL"
        half = self.spread / 2.0
        entry = round(self.last_price + half if side == "BUY" else self.last_price - half, 2)

        unit = self.dynamic_unit
        sl_d = round(unit * 2.1, 2)
        tp_d = round(unit * 3.4, 2)
        sl_price = round(entry - sl_d if side == "BUY" else entry + sl_d, 2)
        tp_price = round(entry + tp_d if side == "BUY" else entry - tp_d, 2)

        conf = decision["score"]
        risk_pct = 0.005 + ((conf - ENTRY_CONFIDENCE) / 100.0) * 0.025
        risk_usd = self.balance * risk_pct
        lots = round(min(2.50, risk_usd / max(1.0, sl_d * 100.0)), 2)
        if lots < 0.01:
            self.cooldown_until = time.time() + 60
            self.log_thought("رُفض الدخول: الحجم الناتج أقل من أدنى لوت (0.01).", "RISK_SHIELD")
            return

        reason = (
            f"قناعة ذاتية {conf}% | السياسة: {self.active_policy} | "
            f"هيرست H={self.hurst} | إنتروبيا={self.entropy}"
        )
        self.open_position = {
            "id": int(time.time() * 1000) % 1000000,
            "side": side,
            "entry_price": entry,
            "current_price": round(self.last_price, 2),
            "sl": sl_price,
            "tp": tp_price,
            "initial_risk": sl_d,
            "risk_usd": round(lots * sl_d * 100.0, 2),
            "lots": lots,
            "initial_lots": lots,
            "realized": 0.0,
            "unrealized_pnl": 0.0,
            "be_secured": False,
            "partial_taken": False,
            "confidence": conf,
            "policy": self.active_policy,
            "regime": self.active_regime,
            "reason_entry": reason,
            "opened_at": stamp(),
            "opened_ts": time.time(),
        }
        self.log_thought(f"دخول مستقل [{side}] عند {entry:.2f}$ بحجم {lots}L — {reason}", "EXECUTION")

    def _manage_open_position(self, decision: Dict):
        pos = self.open_position
        if not pos or self.last_price is None:
            return

        side = pos["side"]
        q = self._exit_quote(side)
        pos["current_price"] = round(self.last_price, 2)

        diff = (q - pos["entry_price"]) if side == "BUY" else (pos["entry_price"] - q)
        floating = round(diff * pos["lots"] * 100.0, 2)
        pos["unrealized_pnl"] = round(floating + pos["realized"], 2)  # إجمالي الصفقة
        self.equity = round(self.balance + floating, 2)

        r_multiple = diff / max(0.1, pos["initial_risk"])
        held = time.time() - pos["opened_ts"]

        # تأمين Break-Even مع جني جزئي
        if not pos["be_secured"] and r_multiple >= 0.75:
            lock = round(pos["initial_risk"] * 0.15, 2)
            pos["sl"] = round(pos["entry_price"] + lock if side == "BUY" else pos["entry_price"] - lock, 2)
            pos["be_secured"] = True

            if not pos["partial_taken"] and pos["lots"] >= 0.10:
                closed_lots = round(pos["lots"] * 0.5, 2)
                chunk = round(diff * closed_lots * 100.0, 2)
                self.balance = round(self.balance + chunk, 2)
                pos["realized"] = round(pos["realized"] + chunk, 2)
                pos["lots"] = round(pos["lots"] - closed_lots, 2)
                pos["partial_taken"] = True
                self.log_thought(
                    f"تأمين ذكي: جني 50% بربح {chunk:+.2f}$ ونقل الوقف لـ {pos['sl']:.2f}$", "RISK_SHIELD"
                )
            else:
                self.log_thought(f"تأمين الصفقة: نقل الوقف إلى {pos['sl']:.2f}$", "RISK_SHIELD")

        # وقف تتبعي
        if pos["be_secured"] and r_multiple >= 1.35:
            trail = round(self.dynamic_unit * 1.1, 2)
            if side == "BUY":
                new_sl = round(q - trail, 2)
                if new_sl > pos["sl"]:
                    pos["sl"] = new_sl
            else:
                new_sl = round(q + trail, 2)
                if new_sl < pos["sl"]:
                    pos["sl"] = new_sl

        # الهدف / الوقف (على سعر الإغلاق الفعلي Bid/Ask)
        hit_tp = q >= pos["tp"] if side == "BUY" else q <= pos["tp"]
        hit_sl = q <= pos["sl"] if side == "BUY" else q >= pos["sl"]
        if hit_tp:
            self._close_position(q, "ضرب الهدف الإحصائي الكامل (Autonomous TP Hit)")
            return
        if hit_sl:
            reason = "وقف الأرباح المتحرك (Secured SL)" if pos["be_secured"] else "ضرب وقف الحماية (Statistical SL)"
            self._close_position(q, reason)
            return

        # خروج استباقي عند إشارة معاكسة قوية
        opposite = (side == "BUY" and decision["direction"] == -1) or (side == "SELL" and decision["direction"] == 1)
        if opposite and decision["score"] >= 80.0 and held >= OPPOSITE_EXIT_MIN_HOLD_S:
            self._close_position(q, f"خروج استباقي: رصد انعكاس زخم مضاد بقوة {decision['score']}%")
            return

        # إلغاء الفرضية عند الركود
        if held > STAGNATION_EXIT_S and abs(r_multiple) < 0.25 and self.entropy > 0.84:
            self._close_position(q, f"إلغاء الفرضية ذاتياً: تباطؤ الزخم وارتفاع الإنتروبيا لـ {self.entropy}")

    def _close_position(self, exit_price: float, reason_exit: str):
        pos = self.open_position
        if not pos:
            return
        self.open_position = None  # أولاً: يمنع الإغلاق المزدوج لو فشل أي خطوة لاحقة

        diff = (exit_price - pos["entry_price"]) if pos["side"] == "BUY" else (pos["entry_price"] - exit_price)
        pnl_rest = round(diff * pos["lots"] * 100.0, 2)
        total_pnl = round(pnl_rest + pos["realized"], 2)  # يشمل الجني الجزئي المحقق سابقاً
        self.balance = round(self.balance + pnl_rest, 2)
        self.equity = self.balance

        record = {
            "id": pos["id"],
            "side": pos["side"],
            "entry_price": pos["entry_price"],
            "exit_price": round(exit_price, 2),
            "lots": pos["initial_lots"],
            "pnl": total_pnl,
            "confidence": pos["confidence"],
            "regime": pos["regime"],
            "reason_entry": pos["reason_entry"],
            "reason_exit": reason_exit,
            "opened_at": pos["opened_at"],
            "closed_at": stamp(),
        }
        self.trade_log.insert(0, record)
        del self.trade_log[50:]

        s = self.stats
        s["total"] += 1
        if total_pnl > 0:
            s["wins"] += 1
            s["gross_win"] += total_pnl
        elif total_pnl < 0:
            s["gross_loss"] += -total_pnl

        # تعلّم الأوزان: متناظر ومقيس بمضاعف المخاطرة R
        pol = pos["policy"]
        if pol in self.policy_weights:
            r_result = total_pnl / max(pos["risk_usd"], 1e-6)
            delta = float(np.clip(r_result, -1.0, 1.5)) * 0.05
            self.policy_weights[pol] = round(float(np.clip(self.policy_weights[pol] + delta, 0.5, 1.8)), 3)

        # فترات التهدئة وحماية سلسلة الخسائر
        now = time.time()
        if total_pnl < 0:
            self.consec_losses += 1
            self.cooldown_until = now + COOLDOWN_AFTER_LOSS_S
        else:
            self.consec_losses = 0
            self.cooldown_until = now + COOLDOWN_AFTER_WIN_S
        if self.consec_losses >= MAX_CONSECUTIVE_LOSSES:
            self.cooldown_until = now + LOSS_STREAK_PAUSE_S
            self.consec_losses = 0
            self.log_thought(
                f"{MAX_CONSECUTIVE_LOSSES} خسائر متتالية: إيقاف الدخول {LOSS_STREAK_PAUSE_S // 60} دقيقة.",
                "RISK_SHIELD",
            )

        try:
            self._persist_trade_and_lesson(record, pol)
        except Exception as e:
            print("Persist error:", repr(e))

        self.log_thought(
            f"إغلاق صفقة [{pos['side']}] عند {exit_price:.2f}$ | الصافي: {total_pnl:+.2f}$ | السبب: {reason_exit}",
            "EXIT_DECISION",
        )

    def _persist_trade_and_lesson(self, rec: Dict, policy_name: str):
        conn = sqlite3.connect(DB_PATH)
        try:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO trade_history (side, entry_price, exit_price, lots, pnl, confidence, regime,
                                           reason_entry, reason_exit, opened_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec["side"], rec["entry_price"], rec["exit_price"], rec["lots"], rec["pnl"],
                    rec["confidence"], rec["regime"], rec["reason_entry"], rec["reason_exit"],
                    rec["opened_at"], rec["closed_at"],
                ),
            )
            c.execute(
                """
                INSERT INTO memory_lessons (timestamp, regime, hurst, entropy, outcome, weight_adjustment, lesson_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec["closed_at"], rec["regime"], self.hurst, self.entropy, rec["pnl"],
                    json.dumps(self.policy_weights),
                    f"تعديل وزن السياسة {policy_name} إلى {self.policy_weights.get(policy_name, 1.0)}",
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_kpi_metrics(self) -> Dict:
        s = self.stats
        total = s["total"]
        win_rate = round(s["wins"] / total * 100.0, 1) if total else 0.0
        if s["gross_loss"] > 0:
            pf = round(s["gross_win"] / s["gross_loss"], 2)
        else:
            pf = None if s["gross_win"] > 0 else 0.0  # None = لا خسائر بعد (∞)
        return {
            "win_rate": win_rate,
            "total_trades": total,
            "profit_factor": pf,
            "net_pnl": round(self.balance - self.initial_balance, 2),
        }


brain = AutonomousQuantitativeBrain()


# ==============================================================================
# 3. محرك بث سعر CFD الذهب الفوري XAU/USD (سعر وسيط حقيقي Bid/Ask — بلا مفتاح وبلا عقود آجلة)
# ==============================================================================
def parse_swissquote(data, profile: str):
    """يستخرج (bid, ask, ts_ms) من رد Swissquote؛ متسامح مع اختلاف الشكل."""
    items = data if isinstance(data, list) else [data]
    bids, asks, stamps = [], [], []
    for it in items:
        if not isinstance(it, dict):
            continue
        row = None
        prices = it.get("spreadProfilePrices")
        if isinstance(prices, list) and prices:
            by = {str(p.get("spreadProfile", "")).lower(): p for p in prices if isinstance(p, dict)}
            row = by.get(profile) or next(iter(by.values()), None)
        elif "bid" in it and "ask" in it:
            row = it
        if not row:
            continue
        try:
            bid, ask = float(row["bid"]), float(row["ask"])
        except (KeyError, TypeError, ValueError):
            continue
        if 100.0 < bid < 100000.0 and 0.0 <= ask - bid < 10.0:
            bids.append(bid)
            asks.append(ask)
            ts = it.get("ts") or it.get("timestamp")
            if isinstance(ts, (int, float)):
                stamps.append(float(ts))
    if not bids:
        return None
    # الوسيط عبر خوادم Swissquote يتجنب القيم الشاذة
    ts_ms = max(stamps) if stamps else None
    if ts_ms is not None and ts_ms < 1e11:  # ثوانٍ لا ميلي ثانية
        ts_ms *= 1000.0
    return float(np.median(bids)), float(np.median(asks)), ts_ms


class LiveGoldStream:
    def __init__(self, brain_ref: AutonomousQuantitativeBrain):
        self.brain = brain_ref
        self.clients: List[WebSocket] = []
        self.feed_source: str = "CONNECTING_TO_LIVE_CFD_FEED"
        self.market_open: bool = True
        self.last_poll: float = 0.0
        self.last_yahoo: float = 0.0
        self.last_good_ts: float = 0.0
        self.last_hist_try: float = 0.0
        self.hist_tries: int = 0
        self.hist_loaded: bool = False
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------- المصادر
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=6.0, headers={"User-Agent": "Mozilla/5.0"})
        return self._client

    async def _quote_swissquote(self):
        try:
            r = await self._http().get(SQ_URL)
            if r.status_code != 200:
                print("Swissquote HTTP", r.status_code)
                return None
            data = r.json()
        except Exception as e:
            print("Swissquote error:", repr(e))
            return None
        parsed = parse_swissquote(data, SQ_PROFILE)
        if parsed is None:
            print("Swissquote parse failed, sample:", str(data)[:300])
        return parsed

    async def _quote_yahoo(self):
        """احتياطي: سعر السبوت (XAUUSD=X) من ياهو — ليس عقوداً آجلة."""
        def work():
            df = yf.Ticker("XAUUSD=X").history(period="1d", interval="1m")
            return round(float(df["Close"].dropna().iloc[-1]), 2)
        try:
            p = await asyncio.get_running_loop().run_in_executor(None, work)
            return p - ASSUMED_SPREAD / 2.0, p + ASSUMED_SPREAD / 2.0, None
        except Exception as e:
            print("Yahoo spot error:", repr(e))
            return None

    async def _load_history(self, ref_price: float):
        """شموع تاريخية لسعر السبوت XAUUSD=X تُزاح لتتطابق مع سعر الوسيط الحي (بلا قفزة)."""
        self.last_hist_try = time.time()
        self.hist_tries += 1

        def work():
            df = yf.Ticker("XAUUSD=X").history(period="5d", interval=YF_INTERVAL).dropna()
            out = []
            for idx, row in df.tail(300).iterrows():
                out.append({
                    "time": int(idx.timestamp()) // TF_SEC * TF_SEC,
                    "open": float(row["Open"]), "high": float(row["High"]),
                    "low": float(row["Low"]), "close": float(row["Close"]),
                })
            return out
        try:
            hist = await asyncio.get_running_loop().run_in_executor(None, work)
        except Exception as e:
            print("History error:", repr(e))
            return
        live = list(self.brain.candles)
        first_live = live[0]["time"] if live else None
        hist = [c for c in hist if first_live is None or c["time"] < first_live]
        # إزالة التكرار في الزمن مع الإبقاء على الترتيب
        seen, clean = set(), []
        for c in hist:
            if c["time"] not in seen:
                seen.add(c["time"])
                clean.append(c)
        if len(clean) < 20:
            return
        anchor = live[0]["open"] if live else ref_price
        off = anchor - clean[-1]["close"]
        if abs(off) > ref_price * 0.03:
            print("History offset too large, ignored:", round(off, 2))
            return
        shifted = [{"time": c["time"], "open": round(c["open"] + off, 2), "high": round(c["high"] + off, 2),
                    "low": round(c["low"] + off, 2), "close": round(c["close"] + off, 2)} for c in clean]
        self.brain.candles.clear()
        self.brain.candles.extend(shifted + live)
        self.brain._analysis_ts = None
        self.hist_loaded = True
        self.brain.log_thought(f"تحميل {len(shifted)} شمعة تاريخية لسعر السبوت (مُطابَقة مع سعر الوسيط).", "PERCEPTION")

    def _update_candle(self, price: float, bucket: int):
        candles = self.brain.candles
        if not candles or bucket > candles[-1]["time"]:
            candles.append({"time": bucket, "open": price, "high": price, "low": price, "close": price})
            return
        last = candles[-1]
        last["close"] = price
        if price > last["high"]:
            last["high"] = price
        if price < last["low"]:
            last["low"] = price

    # --------------------------------------------------------------- الحلقة
    async def run_stream_loop(self):
        try:
            while True:
                try:
                    await self._step()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print("Stream error:", repr(e))
                await asyncio.sleep(1.0)
        finally:
            if self._client is not None:
                await self._client.aclose()

    async def _step(self):
        b = self.brain
        now = time.time()
        bucket = (int(now) // TF_SEC) * TF_SEC

        if now - self.last_poll >= POLL_S:
            self.last_poll = now
            q, src = await self._quote_swissquote(), "SWISSQUOTE"
            if q is None and now - self.last_yahoo >= 15.0:
                self.last_yahoo = now
                q, src = await self._quote_yahoo(), "YAHOO_SPOT"
            if q is not None:
                bid, ask, ts_ms = q
                self.last_good_ts = now
                age = (now - ts_ms / 1000.0) if ts_ms else 0.0
                self.market_open = age <= STALE_QUOTE_S
                self.feed_source = f"LIVE_CFD XAU/USD ({src})"
                if self.market_open:
                    mid = round((bid + ask) / 2.0, 2)
                    b.last_price = mid
                    b.spread = round(max(ask - bid, 0.0), 2)  # السبريد الحقيقي
                    self._update_candle(mid, bucket)
                    if (not self.hist_loaded and self.hist_tries < 5 and now - self.last_hist_try > 30.0):
                        await self._load_history(mid)

        fresh = (now - self.last_good_ts) < 20.0
        was = b.can_trade
        b.can_trade = fresh and self.market_open
        if was and not b.can_trade:
            b.log_thought("توقف البث الحي أو السوق مغلق: لا دخول جديد حتى عودة الأسعار.", "SYSTEM")

        b.process_market_state()
        await self.broadcast_state()

    # ------------------------------------------------------------------ البث
    def feed_label(self) -> str:
        if self.last_good_ts == 0.0:
            return self.feed_source
        if not self.market_open:
            return self.feed_source + " — السوق مغلق"
        if not self.brain.can_trade:
            return self.feed_source + " — STALE"
        return self.feed_source

    def build_payload(self, ptype: str, full: bool = False) -> Dict:
        b = self.brain
        payload = {
            "type": ptype,
            "feed_source": self.feed_label(),
            "bot_active": b.bot_active,
            "price": b.last_price,
            "spread": b.spread,
            "balance": b.balance,
            "equity": b.equity,
            "kpis": b.get_kpi_metrics(),
            "cognition": {
                "hurst": b.hurst,
                "entropy": b.entropy,
                "velocity": b.velocity,
                "sigma": b.volatility_sigma,
                "regime": b.active_regime,
                "policy": b.active_policy,
                "confidence": b.confidence,
                "weights": b.policy_weights,
                "block_reason": b.block_reason,
            },
            "discovered_zones": b.discovered_zones,
            "open_position": b.open_position,
            "trade_log": b.trade_log[:25],
            "thoughts": list(b.thoughts)[:25],
        }
        if full:
            payload["candles"] = list(b.candles)
        else:
            payload["candle"] = b.candles[-1] if b.candles else None
        return payload

    async def _broadcast(self, payload: Dict):
        if not self.clients:
            return

        async def send(ws: WebSocket):
            try:
                await asyncio.wait_for(ws.send_json(payload), timeout=2.0)
                return None
            except Exception:
                return ws

        results = await asyncio.gather(*(send(ws) for ws in list(self.clients)))
        for dead in results:
            if dead is not None and dead in self.clients:
                self.clients.remove(dead)

    async def broadcast_state(self):
        if self.clients:
            await self._broadcast(self.build_payload("TICK_UPDATE"))


stream_manager = LiveGoldStream(brain)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # حفظ مرجع المهمة يمنع جمع القمامة لها بصمت، والإلغاء النظيف عند الإيقاف
    task = asyncio.create_task(stream_manager.run_stream_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Autonomous XAU/USD Quantitative Entity", lifespan=lifespan)


# ==============================================================================
# 4. خادم ومتحكم WebSockets
# ==============================================================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        # إرسال اللقطة الأولية قبل التسجيل في قائمة البث لتفادي الإرسال المتزامن على نفس الاتصال
        await ws.send_json(stream_manager.build_payload("INIT_SNAPSHOT", full=True))
        stream_manager.clients.append(ws)
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except ValueError:
                continue
            cmd = data.get("command") if isinstance(data, dict) else None
            if cmd == "TOGGLE_BOT":
                brain.bot_active = not brain.bot_active
                state_txt = "مُفعّل ومستقل" if brain.bot_active else "مُتوقف مؤقتاً"
                brain.log_thought(f"تم تغيير حالة البوت يدوياً إلى: {state_txt}", "SYSTEM")
            elif cmd == "PANIC_CLOSE" and brain.open_position is not None:
                brain._close_position(
                    brain._exit_quote(brain.open_position["side"]),
                    "إغلاق فوري بطلب المالك (Manual Emergency Exit)",
                )
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WS error:", repr(e))
    finally:
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

        <p id="cog-block" class="text-[11px] text-amber-200 mono leading-relaxed"></p>
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
    let zoneSig = '';
    let posSig = '';

    function syncDiscoveredZones(zones) {
      const sig = JSON.stringify(zones || []);
      if (sig === zoneSig) return;
      zoneSig = sig;
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
      const sig = JSON.stringify(pos ? [pos.entry_price, pos.tp, pos.sl, pos.be_secured, pos.lots, pos.side] : null);
      if (sig === posSig) return;
      posSig = sig;
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

      ws.onopen = () => {
        document.getElementById('ws-dot').className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse';
      };

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

    const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

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
        document.getElementById('kpi-pf').textContent = data.kpis.profit_factor === null ? '∞' : Number(data.kpis.profit_factor).toFixed(2);
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
        document.getElementById('cog-block').textContent = c.block_reason ? '⏸ ' + c.block_reason : '';

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
            <td class="py-1.5 text-slate-300">${esc(t.reason_exit)}</td>
          </tr>
        `).join('');
      }

      if (data.thoughts) {
        const tBox = document.getElementById('thoughts-container');
        tBox.innerHTML = data.thoughts.map(th => `
          <div class="p-2.5 rounded-lg bg-slate-900/75 border border-slate-800/90">
            <div class="flex items-center justify-between text-[10px] text-slate-400 mb-1">
              <span class="font-semibold text-amber-400/90">${esc(th.category)}</span>
              <span class="mono">${th.time}</span>
            </div>
            <p class="text-slate-200 leading-relaxed">${esc(th.text)}</p>
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