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
# 0. إعدادات قابلة للتعديل (Configuration & Hyperparameters)
# ==============================================================================
DB_PATH = os.getenv("BOT_DB_PATH", "autonomous_memory.db")

# --- مصدر السعر: CFD الذهب الفوري XAU/USD من Swissquote (Bid/Ask حقيقي، بلا مفتاح) ---
SQ_URL = "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD"
SQ_PROFILE = os.getenv("SQ_PROFILE", "standard").lower()  # standard | premium | prime
POLL_S = 2.0                     # تكرار جلب السعر (ثوانٍ)
STALE_QUOTE_S = 120              # أكبر عمر مقبول للسعر قبل اعتبار السوق مغلقاً
TF_MIN = int(os.getenv("BOT_TF_MIN", "1"))  # الإطار الزمني بالدقائق: 1 | 5 | 15
_YF = {1: "1m", 5: "5m", 15: "15m"}
if TF_MIN not in _YF:
    raise SystemExit("BOT_TF_MIN must be one of 1, 5, 15")
YF_INTERVAL = _YF[TF_MIN]
TF_SEC = TF_MIN * 60
MIN_CANDLES = 50                 # أقل عدد شموع قبل بدء التحليل
MAX_SPREAD = 2.0                 # لا دخول إذا اتسع السبريد فوق هذا ($)
ASSUMED_SPREAD = 0.25            # سبريد ثابت افتراضي ($) للتداول الورقي عند غياب الفارق
ENTRY_CONFIDENCE = 82.0          # الحد الأدنى للقناعة للدخول
SIGNAL_CONFIRM = 2               # عدد الشموع المغلقة المتتالية لتأكيد الإشارة قبل الدخول
COOLDOWN_AFTER_WIN_S = 120 * TF_MIN
COOLDOWN_AFTER_LOSS_S = 300 * TF_MIN
MAX_CONSECUTIVE_LOSSES = 4
LOSS_STREAK_PAUSE_S = 1800 * TF_MIN
OPPOSITE_EXIT_MIN_HOLD_S = 180 * TF_MIN
STAGNATION_EXIT_S = 900 * TF_MIN


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%m-%d %H:%M:%S")


def get_active_session() -> Dict[str, str]:
    """تحديد جلسة التداول الحالية بتوقيت UTC وتأثيرها على سيولة الذهب."""
    now_utc = datetime.now(timezone.utc)
    h = now_utc.hour + now_utc.minute / 60.0
    if 21.0 <= h < 22.1:
        return {"name": "DAILY_ROLLOVER", "label": "توقف السيولة اليومي (Rollover)", "bias": "HOLD"}
    if 13.0 <= h < 17.0:
        return {"name": "LDN_NY_OVERLAP", "label": "تداخل لندن ونيويورك (ذروة السيولة)", "bias": "MOMENTUM"}
    if 7.0 <= h < 13.0:
        return {"name": "LONDON_SESSION", "label": "جلسة لندن (اندفاع سعري)", "bias": "MOMENTUM"}
    if 17.0 <= h < 21.0:
        return {"name": "NEWYORK_SESSION", "label": "جلسة نيويورك", "bias": "NEUTRAL"}
    return {"name": "ASIAN_SESSION", "label": "الجلسة الآسيوية (ارتداد للمتوسط)", "bias": "REVERSION"}


# ==============================================================================
# 1. الذاكرة الدائمة وسجل الصفقات والحالة النشطة (SQLite Memory Engine)
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
        c.execute("""
            CREATE TABLE IF NOT EXISTS runtime_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


init_db()


# ==============================================================================
# 2. عقل الوكيل الذاتي المستقل (Autonomous Quantitative & SMC Cortex)
# ==============================================================================
class AutonomousQuantitativeBrain:
    POLICIES = (
        "MOMENTUM_CONTINUATION",
        "GRAVITY_MEAN_REVERSION",
        "VOLATILITY_EXPANSION",
    )

    def __init__(self):
        self.lock = asyncio.Lock()
        self.bot_active: bool = True
        self.can_trade: bool = False
        self.lot_mode: str = "FIXED_001"  # FIXED_001 (0.01 ثابت) | DYNAMIC_RISK (نسبة مخاطرة ديناميكية)
        self.balance: float = 10000.0
        self.equity: float = 10000.0
        self.initial_balance: float = 10000.0
        self.peak_balance: float = 10000.0
        self.max_drawdown_pct: float = 0.0

        self.candles: deque = deque(maxlen=400)
        self.last_price: Optional[float] = None
        self.bid_price: Optional[float] = None
        self.ask_price: Optional[float] = None
        self.spread: float = ASSUMED_SPREAD

        # المؤشرات الكمية والهيكلية
        self.hurst: float = 0.50
        self.entropy: float = 0.50
        self.velocity: float = 0.0
        self.volatility_sigma: float = 0.0
        self.atr_value: float = 1.20
        self.z_score: float = 0.0
        self.poc_price: Optional[float] = None
        self.vah_price: Optional[float] = None
        self.val_price: Optional[float] = None
        self.smc_event: str = "NONE"

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
        self.thoughts: deque = deque(maxlen=50)
        self.trade_log: List[Dict] = []
        self.stats = {"total": 0, "wins": 0, "gross_win": 0.0, "gross_loss": 0.0}
        self.consec_losses: int = 0
        self.cooldown_until: float = 0.0

        self._analysis_ts: Optional[int] = None
        self._decision: Dict = {"policy": "CALIBRATING", "direction": 0, "score": 0.0}
        self._restore_state()

    # ----------------------------------------------------------------- الذاكرة
    def _save_runtime_key(self, key: str, val_obj):
        try:
            conn = sqlite3.connect(DB_PATH)
            try:
                c = conn.cursor()
                c.execute(
                    "INSERT OR REPLACE INTO runtime_state (key, value) VALUES (?, ?)",
                    (key, json.dumps(val_obj)),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            print("Runtime save error:", repr(e))

    def _restore_state(self):
        """استرجاع الرصيد، الأوزان، الإحصائيات، والصفقة المفتوحة من SQLite بعد إعادة التشغيل."""
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
                self.peak_balance = max(self.initial_balance, self.balance)

                c.execute("SELECT weight_adjustment FROM memory_lessons ORDER BY id DESC LIMIT 1")
                row = c.fetchone()
                if row and row[0]:
                    saved = json.loads(row[0])
                    for k in self.policy_weights:
                        if k in saved:
                            self.policy_weights[k] = float(np.clip(float(saved[k]), 0.5, 1.8))

                # استرجاع الصفقة المفتوحة ووضع اللوت وفترة التهدئة
                c.execute("SELECT key, value FROM runtime_state")
                for k, v in c.fetchall():
                    if k == "open_position" and v:
                        pos = json.loads(v)
                        if isinstance(pos, dict) and "side" in pos:
                            self.open_position = pos
                    elif k == "lot_mode" and v:
                        mode = json.loads(v)
                        if mode in ("FIXED_001", "DYNAMIC_RISK"):
                            self.lot_mode = mode
                    elif k == "cooldown_until" and v:
                        self.cooldown_until = float(json.loads(v))
            finally:
                conn.close()
        except Exception as e:
            print("Restore error:", repr(e))

        streak = 0
        for t in self.trade_log:
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

    # --------------------------------------------------------------- الإحصاء والرياضيات الكمية
    def compute_hurst_exponent(self, prices: np.ndarray) -> float:
        """حساب معامل هيرست المصحح باستخدام الفروق اللوغاريتمية المُزال منها الانجراف (Detrended Log-Returns)."""
        if len(prices) < 45:
            return 0.5
        log_p = np.log(np.maximum(prices, 1e-4))
        lags = range(2, min(25, len(log_p) // 2))
        tau, valid_lags = [], []
        for lag in lags:
            diff = log_p[lag:] - log_p[:-lag]
            # إزالة الانجراف المتوسط لمنع انحياز الشمعة الواحدة
            diff_detrended = diff - np.mean(diff)
            std_val = float(np.std(diff_detrended))
            if std_val > 1e-8:
                tau.append(std_val)
                valid_lags.append(lag)
        if len(valid_lags) < 4:
            return 0.5
        poly = np.polyfit(np.log(valid_lags), np.log(tau), 1)
        return round(float(np.clip(poly[0], 0.15, 0.85)), 3)

    def compute_shannon_entropy(self, returns: np.ndarray) -> float:
        """حساب إنتروبيا شانون الحقيقية عبر التعداد الاحتمالي المباشر (density=False) مع تنعيم لابلاس."""
        if len(returns) < 20:
            return 0.75
        std_r = float(np.std(returns))
        if std_r <= 1e-8:
            return 0.30
        num_bins = 10
        counts, _ = np.histogram(returns, bins=num_bins, density=False)
        # تنعيم لابلاس الطفيف لمنع القفزات الحادة عند الفئات الفارغة
        smoothed = counts.astype(float) + 0.05
        probs = smoothed / np.sum(smoothed)
        entropy = -np.sum(probs * np.log2(probs)) / np.log2(num_bins)
        return round(float(np.clip(entropy, 0.05, 0.99)), 3)

    def compute_true_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """حساب متوسط المدى الحقيقي (Wilder's True Range ATR) شاملة الفجوات السعرية."""
        if len(closes) < 2:
            return 1.2
        prev_closes = closes[:-1]
        cur_highs = highs[1:]
        cur_lows = lows[1:]
        tr = np.maximum(
            cur_highs - cur_lows,
            np.maximum(np.abs(cur_highs - prev_closes), np.abs(cur_lows - prev_closes)),
        )
        n = min(period, len(tr))
        atr = float(np.mean(tr[-n:]))
        if atr <= 0.08:
            atr = max(0.8, float(closes[-1]) * 0.0004)
        return round(atr, 3)

    def discover_gravity_and_volume_profile(
        self, opens: np.ndarray, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray
    ):
        """اكتشاف مناطق الجاذبية/الرفض + حساب Volume Profile التقديري (POC, VAH, VAL)."""
        if len(closes) < 50:
            return
        min_p, max_p = float(np.min(lows)), float(np.max(highs))
        span = max_p - min_p
        if span <= 1e-5:
            return

        grid = np.linspace(min_p, max_p, 40)
        bandwidth = span / 22.0
        densities = np.zeros_like(grid)
        rejection_energy = np.zeros_like(grid)
        vol_profile = np.zeros_like(grid)

        wicks = (highs - np.maximum(opens, closes)) + (np.minimum(opens, closes) - lows)
        ranges = np.maximum(highs - lows, 0.05)

        for i, level in enumerate(grid):
            dist = np.abs(closes - level)
            kernel = np.exp(-0.5 * (dist / bandwidth) ** 2)
            densities[i] = np.sum(kernel)
            # بروفيل الحجم المرجح بتواجد السعر ونشاط الشمعة
            in_candle = (lows <= level + bandwidth * 0.5) & (highs >= level - bandwidth * 0.5)
            if np.any(in_candle):
                vol_profile[i] = float(np.sum(ranges[in_candle]))
            near_mask = dist < (bandwidth * 1.2)
            if np.any(near_mask):
                rejection_energy[i] = float(np.mean(wicks[near_mask]))

        # استخراج POC و VAH و VAL (منطقة القيمة 70%)
        poc_idx = int(np.argmax(vol_profile))
        self.poc_price = round(float(grid[poc_idx]), 2)
        total_vp = float(np.sum(vol_profile)) + 1e-9
        accum = float(vol_profile[poc_idx])
        lo_i, hi_i = poc_idx, poc_idx
        while accum / total_vp < 0.70 and (lo_i > 0 or hi_i < len(grid) - 1):
            next_lo = vol_profile[lo_i - 1] if lo_i > 0 else -1.0
            next_hi = vol_profile[hi_i + 1] if hi_i < len(grid) - 1 else -1.0
            if next_hi >= next_lo and hi_i < len(grid) - 1:
                hi_i += 1
                accum += float(vol_profile[hi_i])
            elif lo_i > 0:
                lo_i -= 1
                accum += float(vol_profile[lo_i])
            else:
                break
        self.val_price = round(float(grid[lo_i]), 2)
        self.vah_price = round(float(grid[hi_i]), 2)

        d_norm = densities / (np.max(densities) + 1e-9)
        r_norm = rejection_energy / (np.max(rejection_energy) + 1e-9)
        combined = d_norm + 0.65 * r_norm
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

    def detect_smc_microstructure(
        self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> Dict:
        """رصد هياكل SMC للذهب: سحب السيولة (Liquidity Sweep) والفجوات السعرية العادلة (FVG)."""
        result = {"bull_sweep": False, "bear_sweep": False, "bull_fvg": False, "bear_fvg": False, "label": "NONE"}
        if len(closes) < 25:
            return result

        # آخر شمعة مغلقة
        o1, h1, l1, c1 = opens[-1], highs[-1], lows[-1], closes[-1]
        rng1 = max(h1 - l1, 1e-4)
        lower_wick = min(o1, c1) - l1
        upper_wick = h1 - max(o1, c1)

        # القاع والقمة المحلية للشموع الـ 15 السابقة لها
        prev_low = float(np.min(lows[-16:-1]))
        prev_high = float(np.max(highs[-16:-1]))

        # Liquidity Sweep صاعد: كسر القاع المحلي بذيل سفلي يمثل أكثر من 40% من الشمعة والإغلاق فوق القاع
        if l1 < prev_low and c1 > prev_low and (lower_wick / rng1) >= 0.40:
            result["bull_sweep"] = True
            result["label"] = "BULL_LIQUIDITY_SWEEP"
        # Liquidity Sweep هابط: كسر القمة المحلية بذيل علوي يمثل أكثر من 40% من الشمعة والإغلاق تحت القمة
        elif h1 > prev_high and c1 < prev_high and (upper_wick / rng1) >= 0.40:
            result["bear_sweep"] = True
            result["label"] = "BEAR_LIQUIDITY_SWEEP"

        # Fair Value Gap (FVG) على آخر 3 شموع مغلقة
        if lows[-1] > highs[-3] and (closes[-2] - opens[-2]) > self.atr_value * 0.55:
            result["bull_fvg"] = True
            if result["label"] == "NONE":
                result["label"] = "BULLISH_FVG_DISPLACEMENT"
        elif highs[-1] < lows[-3] and (opens[-2] - closes[-2]) > self.atr_value * 0.55:
            result["bear_fvg"] = True
            if result["label"] == "NONE":
                result["label"] = "BEARISH_FVG_DISPLACEMENT"

        self.smc_event = result["label"]
        return result

    def synthesize_optimal_policy(
        self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> Dict:
        returns = np.diff(closes)
        if len(returns) < 35:
            self.active_regime = "CALIBRATING"
            return {"policy": "CALIBRATING", "direction": 0, "score": 0.0, "reason": "معايرة: بيانات غير كافية"}

        session_info = get_active_session()
        if session_info["bias"] == "HOLD":
            self.active_regime = "DAILY_ROLLOVER_PAUSE"
            return {
                "policy": "ROLLOVER_SHIELD",
                "direction": 0,
                "score": 20.0,
                "reason": "توقف السيولة اليومي (21:00-22:05 UTC) — حماية من اتساع السبريد",
            }

        sigma = float(np.std(returns)) + 1e-6
        self.volatility_sigma = round(sigma, 3)

        short_drift = float(np.mean(returns[-6:])) / sigma
        med_drift = float(np.mean(returns[-20:])) / sigma
        z_score = float((closes[-1] - np.mean(closes[-35:])) / (np.std(closes[-35:]) + 1e-6))
        self.z_score = round(z_score, 2)

        smc = self.detect_smc_microstructure(opens, highs, lows, closes)

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

        def weighted(policy: str, s: float) -> float:
            w = self.policy_weights[policy]
            # تعزيز وزن السياسة المتوافقة مع الجلسة النشطة
            if session_info["bias"] == "MOMENTUM" and policy in ("MOMENTUM_CONTINUATION", "VOLATILITY_EXPANSION"):
                w *= 1.08
            elif session_info["bias"] == "REVERSION" and policy == "GRAVITY_MEAN_REVERSION":
                w *= 1.08
            return s * w if s >= 0 else s / w

        scores = {
            "MOMENTUM_CONTINUATION": weighted("MOMENTUM_CONTINUATION", mom_score),
            "GRAVITY_MEAN_REVERSION": weighted("GRAVITY_MEAN_REVERSION", rev_score),
            "VOLATILITY_EXPANSION": weighted("VOLATILITY_EXPANSION", brk_score),
        }

        if len(self.hist_entropy) >= 60 and self.entropy_rank > 0.90:
            self.active_regime = "HIGH_ENTROPY_NOISE"
            return {
                "policy": "NOISE_FILTER_HOLD",
                "direction": 0,
                "score": 32.0,
                "reason": f"ضجيج عشوائي مرتفع (إنتروبيا {self.entropy} ضمن أعلى 10%)",
            }

        if self.hurst_rank > 0.62:
            candidates = ("MOMENTUM_CONTINUATION", "VOLATILITY_EXPANSION")
            regime_name = "PERSISTENT_TREND"
        elif self.hurst_rank < 0.38:
            candidates = ("GRAVITY_MEAN_REVERSION",)
            regime_name = "MEAN_REVERTING_FIELD"
        else:
            candidates = tuple(scores.keys())
            regime_name = "BALANCED_HYBRID"

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
                conviction = 74.0 + min(20.0, abs(short_drift) * 13.5 + max(0.0, raw_edge * 2.4))
                # تأكيد FVG يعزز قناعة الزخم
                if (direction == 1 and smc["bull_fvg"]) or (direction == -1 and smc["bear_fvg"]):
                    conviction += 5.5
            else:
                conviction = 56.0 + min(15.0, abs(short_drift) * 10.0)
        else:
            # الارتداد للمتوسط مع دعم سحب السيولة (SMC Sweep)
            if (z_score < -1.20 and short_drift > -0.18) or smc["bull_sweep"]:
                direction = 1
            elif (z_score > 1.20 and short_drift < 0.18) or smc["bear_sweep"]:
                direction = -1
            if direction != 0:
                conviction = 75.5 + min(19.0, abs(z_score) * 7.2 + max(0.0, raw_edge * 2.4))
                if (direction == 1 and smc["bull_sweep"]) or (direction == -1 and smc["bear_sweep"]):
                    conviction += 6.5
            else:
                conviction = 54.0 + min(18.0, abs(z_score) * 8.0)

        if direction != 0 and raw_edge <= 0 and self.smc_event == "NONE":
            direction = 0
            conviction = min(conviction, 70.0)
            reason = "السياسة بلا ميزة إحصائية حديثة أو هيكل SMC"

        # تقييم موقع السعر بالنسبة لمناطق الشذوذ والـ POC
        if self.discovered_zones and direction != 0:
            cur_p = self.last_price if self.last_price is not None else float(closes[-1])
            nearest = min(self.discovered_zones, key=lambda z: abs(z["price"] - cur_p))
            gap = nearest["price"] - cur_p
            if abs(gap) < self.atr_value * 2.5:
                behind = (gap <= 0) if direction == 1 else (gap >= 0)
                conviction += 6.0 if behind else -7.5

        final = round(float(np.clip(conviction, 15.0, 97.5)), 1)
        if direction == 0 and not reason:
            reason = "لا إعداد: بانتظار اندفاع سعري أو FVG" if trend_mode else "لا إعداد: السعر داخل منطقة التوازن"
        elif direction != 0 and final < ENTRY_CONFIDENCE:
            reason = f"إعداد قيد التكوين لكن القناعة {final}% < {ENTRY_CONFIDENCE}%"
        elif direction != 0:
            reason = f"إشارة مكتملة ({self.smc_event if self.smc_event != 'NONE' else policy})"

        return {
            "policy": policy,
            "direction": direction,
            "score": final,
            "reason": reason,
            "z_score": round(z_score, 2),
            "short_drift": round(short_drift, 2),
            "smc_event": self.smc_event,
        }

    # ----------------------------------------------------------- دورة القرار
    @staticmethod
    def _rank(history, x: float) -> float:
        arr = np.asarray(history, dtype=float)
        return float(np.mean(arr <= x)) if len(arr) else 0.5

    def _warm_history(self, closes: np.ndarray):
        """تصفير وتعبئة سجل هيرست/الإنتروبيا بنظافة من التاريخ المدمج."""
        self.hist_hurst.clear()
        self.hist_entropy.clear()
        start = max(91, len(closes) - 300)
        for j in range(start, len(closes)):
            self.hist_hurst.append(self.compute_hurst_exponent(closes[j - 90:j]))
            self.hist_entropy.append(self.compute_shannon_entropy(np.diff(closes[max(0, j - 61):j])))

    def _analyze(self, closed: List[Dict]):
        """تحليل كمّي وهيكلي على الشموع المغلقة فقط — يُنفَّذ مرة واحدة عند إغلاق كل شمعة."""
        opens = np.array([c["open"] for c in closed], dtype=float)
        closes = np.array([c["close"] for c in closed], dtype=float)
        highs = np.array([c["high"] for c in closed], dtype=float)
        lows = np.array([c["low"] for c in closed], dtype=float)
        returns = np.diff(closes)

        if len(self.hist_hurst) < 60 and len(closes) >= 160:
            self._warm_history(closes)

        self.atr_value = self.compute_true_atr(highs, lows, closes, period=14)
        self.hurst = self.compute_hurst_exponent(closes[-90:])
        self.entropy = self.compute_shannon_entropy(returns[-60:])
        self.hist_hurst.append(self.hurst)
        self.hist_entropy.append(self.entropy)
        self.hurst_rank = self._rank(self.hist_hurst, self.hurst)
        self.entropy_rank = self._rank(self.hist_entropy, self.entropy)
        self.velocity = round(float(np.mean(returns[-5:])), 3)

        self.discover_gravity_and_volume_profile(opens[-140:], closes[-140:], highs[-140:], lows[-140:])

        self._decision = self.synthesize_optimal_policy(opens, highs, lows, closes)
        self.active_policy = self._decision["policy"]
        self.confidence = self._decision["score"]

        d = self._decision
        if d["direction"] != 0 and d["score"] >= ENTRY_CONFIDENCE:
            # إذا وُجد حدث SMC صريح (Liquidity Sweep أو FVG) مع قناعة >= 87%، يكفي تأكيد شمعة واحدة
            req_confirm = 1 if (d.get("smc_event", "NONE") != "NONE" and d["score"] >= 87.0) else SIGNAL_CONFIRM
            self._streak = self._streak + 1 if d["direction"] == self._last_dir else 1
            self._last_dir = d["direction"]
            d["confirmed"] = self._streak >= req_confirm
            if not d["confirmed"]:
                d["reason"] = f"إشارة أولية — بانتظار تأكيد الإغلاق ({self._streak}/{req_confirm} شموع)"
        else:
            self._streak, self._last_dir = 0, 0
            d["confirmed"] = False

    async def process_market_state(self):
        async with self.lock:
            if len(self.candles) < MIN_CANDLES or self.last_price is None:
                self.block_reason = f"تجميع البيانات ({len(self.candles)}/{MIN_CANDLES} شمعة)"
                return

            forming_ts = self.candles[-1]["time"]
            if forming_ts != self._analysis_ts:
                self._analysis_ts = forming_ts
                self._analyze(list(self.candles)[:-1])

            decision = self._decision

            if self.open_position is not None:
                self.block_reason = "صفقة مفتوحة قيد الإدارة"
                self._manage_open_position(decision)
                return

            self.equity = self.balance
            if not self.bot_active:
                self.block_reason = "الوكيل متوقف يدوياً"
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
                self.block_reason = "إشارة مؤكدة — جاري التنفيذ"
                self._open_position(decision)
            else:
                self.block_reason = decision.get("reason", "لا إعداد")

    def _exit_quote(self, side: str) -> float:
        half = self.spread / 2.0
        p = self.last_price - half if side == "BUY" else self.last_price + half
        return round(p, 2)

    def _compute_structural_stops(self, side: str, entry: float) -> tuple[float, float, float]:
        """حساب الوقف والهدف بناءً على هيكل الشموع والمناطق المكتشفة و ATR الحقيقي."""
        closed = list(self.candles)[:-1]
        recent = closed[-12:] if len(closed) >= 12 else closed
        atr = max(0.6, self.atr_value)

        if side == "BUY":
            swing_low = min(c["low"] for c in recent) if recent else entry - atr * 2.0
            raw_sl = min(entry - atr * 1.6, swing_low - atr * 0.35)
            sl_d = float(np.clip(entry - raw_sl, atr * 1.4, atr * 3.2))
            tp_d = round(sl_d * 1.75, 2)
            # إذا كانت هناك منطقة جاذبية/مقاومة قوية أمام الهدف قريباً، نضع الهدف قبلها بقليل
            above_zones = [z["price"] for z in self.discovered_zones if z["price"] > entry + sl_d * 1.1]
            if above_zones:
                nearest_res = min(above_zones)
                if nearest_res - entry < tp_d * 1.15:
                    tp_d = max(round(sl_d * 1.35, 2), round(nearest_res - entry - 0.15, 2))
            sl_d = round(sl_d, 2)
            return round(entry - sl_d, 2), round(entry + tp_d, 2), sl_d
        else:
            swing_high = max(c["high"] for c in recent) if recent else entry + atr * 2.0
            raw_sl = max(entry + atr * 1.6, swing_high + atr * 0.35)
            sl_d = float(np.clip(raw_sl - entry, atr * 1.4, atr * 3.2))
            tp_d = round(sl_d * 1.75, 2)
            below_zones = [z["price"] for z in self.discovered_zones if z["price"] < entry - sl_d * 1.1]
            if below_zones:
                nearest_sup = max(below_zones)
                if entry - nearest_sup < tp_d * 1.15:
                    tp_d = max(round(sl_d * 1.35, 2), round(entry - nearest_sup - 0.15, 2))
            sl_d = round(sl_d, 2)
            return round(entry + sl_d, 2), round(entry - tp_d, 2), sl_d

    def _open_position(self, decision: Dict):
        if self.balance <= 0 or self.last_price is None:
            return
        side = "BUY" if decision["direction"] == 1 else "SELL"
        half = self.spread / 2.0
        entry = round(self.last_price + half if side == "BUY" else self.last_price - half, 2)

        sl_price, tp_price, sl_d = self._compute_structural_stops(side, entry)
        conf = decision["score"]

        if self.lot_mode == "FIXED_001":
            lots = 0.01
        else:
            risk_pct = 0.005 + ((conf - ENTRY_CONFIDENCE) / 100.0) * 0.020
            risk_usd = self.balance * risk_pct
            lots = round(min(2.50, risk_usd / max(1.0, sl_d * 100.0)), 2)

        if lots < 0.01:
            self.cooldown_until = time.time() + 60
            self.log_thought("رُفض الدخول: الحجم المحسوب أقل من 0.01 لوت.", "RISK_SHIELD")
            return

        smc_tag = f" | SMC={self.smc_event}" if self.smc_event != "NONE" else ""
        reason = (
            f"قناعة {conf}% | السياسة: {self.active_policy}{smc_tag} | "
            f"H={self.hurst} | ATR={self.atr_value}$"
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
            "r_multiple": 0.0,
            "be_secured": False,
            "partial_taken": False,
            "confidence": conf,
            "policy": self.active_policy,
            "regime": self.active_regime,
            "reason_entry": reason,
            "opened_at": stamp(),
            "opened_ts": time.time(),
        }
        self._save_runtime_key("open_position", self.open_position)
        self.log_thought(f"دخول [{side}] عند {entry:.2f}$ بحجم {lots}L (SL={sl_price}$, TP={tp_price}$) — {reason}", "EXECUTION")

    def _manage_open_position(self, decision: Dict):
        pos = self.open_position
        if not pos or self.last_price is None:
            return

        side = pos["side"]
        q = self._exit_quote(side)
        pos["current_price"] = round(self.last_price, 2)

        diff = (q - pos["entry_price"]) if side == "BUY" else (pos["entry_price"] - q)
        floating = round(diff * pos["lots"] * 100.0, 2)
        pos["unrealized_pnl"] = round(floating + pos["realized"], 2)
        self.equity = round(self.balance + floating, 2)

        r_multiple = diff / max(0.1, pos["initial_risk"])
        pos["r_multiple"] = round(r_multiple, 2)
        held = time.time() - pos["opened_ts"]

        # فصل منطق التأمين: للوت >= 0.02 جني جزئي عند 0.85R، وللوت 0.01 تأمين Break-Even عند 1.15R لتفادي الذيول الكاذبة
        can_partial = pos["lots"] >= 0.02
        be_trigger_r = 0.85 if can_partial else 1.15

        if not pos["be_secured"] and r_multiple >= be_trigger_r:
            lock = round(max(self.spread + 0.05, pos["initial_risk"] * 0.18), 2)
            pos["sl"] = round(pos["entry_price"] + lock if side == "BUY" else pos["entry_price"] - lock, 2)
            pos["be_secured"] = True

            if not pos["partial_taken"] and can_partial:
                closed_lots = round(max(0.01, round(pos["lots"] * 0.5, 2)), 2)
                if pos["lots"] - closed_lots >= 0.01:
                    chunk = round(diff * closed_lots * 100.0, 2)
                    self.balance = round(self.balance + chunk, 2)
                    pos["realized"] = round(pos["realized"] + chunk, 2)
                    pos["lots"] = round(pos["lots"] - closed_lots, 2)
                    pos["partial_taken"] = True
                    self.log_thought(
                        f"تأمين ذكي: جني جزئي ({closed_lots}L) بربح {chunk:+.2f}$ ونقل الوقف لـ {pos['sl']:.2f}$",
                        "RISK_SHIELD",
                    )
            else:
                self.log_thought(
                    f"تأمين Break-Even+ عند {r_multiple:.2f}R: نقل الوقف إلى {pos['sl']:.2f}$",
                    "RISK_SHIELD",
                )
            self._save_runtime_key("open_position", pos)

        # وقف تتبعي محمي بحد أدنى مرتبط بالمخاطرة الأولية و ATR الحقيقي
        if pos["be_secured"] and r_multiple >= 1.40:
            trail_dist = round(max(pos["initial_risk"] * 0.55, self.atr_value * 1.15), 2)
            if side == "BUY":
                new_sl = round(q - trail_dist, 2)
                if new_sl > pos["sl"]:
                    pos["sl"] = new_sl
                    self._save_runtime_key("open_position", pos)
            else:
                new_sl = round(q + trail_dist, 2)
                if new_sl < pos["sl"]:
                    pos["sl"] = new_sl
                    self._save_runtime_key("open_position", pos)

        hit_tp = q >= pos["tp"] if side == "BUY" else q <= pos["tp"]
        hit_sl = q <= pos["sl"] if side == "BUY" else q >= pos["sl"]
        if hit_tp:
            self._close_position(q, "ضرب الهدف الإحصائي الهيكلي (Structural TP Hit)")
            return
        if hit_sl:
            reason = "وقف الأرباح المتحرك (Secured Trailing SL)" if pos["be_secured"] else "ضرب وقف الحماية الهيكلي (Structural SL)"
            self._close_position(q, reason)
            return

        opposite = (side == "BUY" and decision["direction"] == -1) or (side == "SELL" and decision["direction"] == 1)
        if opposite and decision["score"] >= 82.0 and held >= OPPOSITE_EXIT_MIN_HOLD_S:
            self._close_position(q, f"خروج استباقي: انعكاس هيكلي مضاد بقوة {decision['score']}%")
            return

        if held > STAGNATION_EXIT_S and abs(r_multiple) < 0.25 and self.entropy > 0.84:
            self._close_position(q, f"إلغاء الفرضية ذاتياً: تباطؤ الزخم وارتفاع الإنتروبيا لـ {self.entropy}")

    def _close_position(self, exit_price: float, reason_exit: str):
        pos = self.open_position
        if not pos:
            return
        self.open_position = None
        self._save_runtime_key("open_position", None)

        diff = (exit_price - pos["entry_price"]) if pos["side"] == "BUY" else (pos["entry_price"] - exit_price)
        pnl_rest = round(diff * pos["lots"] * 100.0, 2)
        total_pnl = round(pnl_rest + pos["realized"], 2)
        self.balance = round(self.balance + pnl_rest, 2)
        self.equity = self.balance

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        dd_pct = round((self.peak_balance - self.balance) / max(1.0, self.peak_balance) * 100.0, 2)
        if dd_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = dd_pct

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

        pol = pos["policy"]
        if pol in self.policy_weights:
            r_result = total_pnl / max(pos["risk_usd"], 1e-6)
            delta = float(np.clip(r_result, -1.0, 1.5)) * 0.05
            self.policy_weights[pol] = round(float(np.clip(self.policy_weights[pol] + delta, 0.5, 1.8)), 3)

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
                f"{MAX_CONSECUTIVE_LOSSES} خسائر متتالية: تفعيل درع الحماية وإيقاف الدخول {LOSS_STREAK_PAUSE_S // 60} دقيقة.",
                "RISK_SHIELD",
            )
        self._save_runtime_key("cooldown_until", self.cooldown_until)

        try:
            self._persist_trade_and_lesson(record, pol)
        except Exception as e:
            print("Persist error:", repr(e))

        self.log_thought(
            f"إغلاق [{pos['side']}] عند {exit_price:.2f}$ | الصافي: {total_pnl:+.2f}$ | السبب: {reason_exit}",
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
            pf = None if s["gross_win"] > 0 else 0.0
        net_pnl = round(self.balance - self.initial_balance, 2)
        expectancy = round(net_pnl / total, 2) if total else 0.0
        return {
            "win_rate": win_rate,
            "total_trades": total,
            "wins": s["wins"],
            "losses": total - s["wins"],
            "profit_factor": pf,
            "net_pnl": net_pnl,
            "expectancy": expectancy,
            "max_drawdown_pct": self.max_drawdown_pct,
        }


brain = AutonomousQuantitativeBrain()


# ==============================================================================
# 3. محرك بث سعر CFD الذهب الفوري XAU/USD
# ==============================================================================
def parse_swissquote(data, profile: str):
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
    ts_ms = max(stamps) if stamps else None
    if ts_ms is not None and ts_ms < 1e11:
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

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=6.0, headers={"User-Agent": "Mozilla/5.0"})
        return self._client

    async def _quote_swissquote(self):
        try:
            r = await self._http().get(SQ_URL)
            if r.status_code != 200:
                return None
            data = r.json()
        except Exception:
            return None
        return parse_swissquote(data, SQ_PROFILE)

    async def _quote_yahoo(self):
        def work():
            df = yf.Ticker("XAUUSD=X").history(period="1d", interval="1m")
            return round(float(df["Close"].dropna().iloc[-1]), 2)
        try:
            p = await asyncio.get_running_loop().run_in_executor(None, work)
            return p - ASSUMED_SPREAD / 2.0, p + ASSUMED_SPREAD / 2.0, None
        except Exception:
            return None

    async def _load_history(self, ref_price: float):
        self.last_hist_try = time.time()
        self.hist_tries += 1

        def work():
            df = yf.Ticker("XAUUSD=X").history(period="5d", interval=YF_INTERVAL).dropna()
            out = []
            for idx, row in df.tail(320).iterrows():
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

        async with self.brain.lock:
            live = list(self.brain.candles)
            first_live = live[0]["time"] if live else None
            hist = [c for c in hist if first_live is None or c["time"] < first_live]
            seen, clean = set(), []
            for c in hist:
                if c["time"] not in seen:
                    seen.add(c["time"])
                    clean.append(c)
            if len(clean) < 25:
                return
            anchor = live[0]["open"] if live else ref_price
            off = anchor - clean[-1]["close"]
            if abs(off) > ref_price * 0.03:
                return
            shifted = [
                {
                    "time": c["time"],
                    "open": round(c["open"] + off, 2),
                    "high": round(c["high"] + off, 2),
                    "low": round(c["low"] + off, 2),
                    "close": round(c["close"] + off, 2),
                }
                for c in clean
            ]
            self.brain.candles.clear()
            self.brain.candles.extend(shifted + live)
            # تصفير وإعادة بناء السجل التاريخي لهيرست والإنتروبيا بعد الدمج
            closes_arr = np.array([c["close"] for c in self.brain.candles], dtype=float)
            self.brain._warm_history(closes_arr)
            self.brain._analysis_ts = None
            self.hist_loaded = True
            self.brain.log_thought(
                f"تم دمج ومعايرة {len(shifted)} شمعة تاريخية لسعر السبوت بنجاح.",
                "PERCEPTION",
            )

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
                    b.bid_price = round(bid, 2)
                    b.ask_price = round(ask, 2)
                    b.last_price = mid
                    b.spread = round(max(ask - bid, 0.0), 2)
                    self._update_candle(mid, bucket)
                    if not self.hist_loaded and self.hist_tries < 5 and now - self.last_hist_try > 30.0:
                        await self._load_history(mid)

        fresh = (now - self.last_good_ts) < 20.0
        was = b.can_trade
        b.can_trade = fresh and self.market_open
        if was and not b.can_trade:
            b.log_thought("توقف البث الحي أو السوق مغلق: تم تفعيل قفل الأمان.", "SYSTEM")

        await b.process_market_state()
        await self.broadcast_state()

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
        now_sec = int(time.time())
        candle_rem_s = TF_SEC - (now_sec % TF_SEC)
        payload = {
            "type": ptype,
            "tf_min": TF_MIN,
            "candle_rem_s": candle_rem_s,
            "feed_source": self.feed_label(),
            "session": get_active_session(),
            "bot_active": b.bot_active,
            "lot_mode": b.lot_mode,
            "price": b.last_price,
            "bid": b.bid_price,
            "ask": b.ask_price,
            "spread": b.spread,
            "balance": b.balance,
            "equity": b.equity,
            "kpis": b.get_kpi_metrics(),
            "cognition": {
                "hurst": b.hurst,
                "hurst_rank": round(b.hurst_rank * 100, 1),
                "entropy": b.entropy,
                "entropy_rank": round(b.entropy_rank * 100, 1),
                "velocity": b.velocity,
                "sigma": b.volatility_sigma,
                "atr": b.atr_value,
                "z_score": b.z_score,
                "poc": b.poc_price,
                "vah": b.vah_price,
                "val": b.val_price,
                "smc_event": b.smc_event,
                "regime": b.active_regime,
                "policy": b.active_policy,
                "confidence": b.confidence,
                "weights": b.policy_weights,
                "block_reason": b.block_reason,
            },
            "discovered_zones": b.discovered_zones,
            "open_position": b.open_position,
            "trade_log": b.trade_log[:30],
            "thoughts": list(b.thoughts)[:35],
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
    task = asyncio.create_task(stream_manager.run_stream_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Autonomous XAU/USD Quantitative React Terminal", lifespan=lifespan)


# ==============================================================================
# 4. خادم ومتحكم WebSockets
# ==============================================================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        await ws.send_json(stream_manager.build_payload("INIT_SNAPSHOT", full=True))
        stream_manager.clients.append(ws)
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except ValueError:
                continue
            cmd = data.get("command") if isinstance(data, dict) else None
            async with brain.lock:
                if cmd == "TOGGLE_BOT":
                    brain.bot_active = not brain.bot_active
                    state_txt = "مُفعّل ومستقل" if brain.bot_active else "مُتوقف مؤقتاً"
                    brain.log_thought(f"تم تغيير حالة الوكيل يدوياً إلى: {state_txt}", "SYSTEM")
                elif cmd == "TOGGLE_LOT_MODE":
                    brain.lot_mode = "DYNAMIC_RISK" if brain.lot_mode == "FIXED_001" else "FIXED_001"
                    brain._save_runtime_key("lot_mode", brain.lot_mode)
                    mode_lbl = "0.01 لوت ثابت" if brain.lot_mode == "FIXED_001" else "مخاطرة ديناميكية %"
                    brain.log_thought(f"تم تبديل نظام حجم العقد إلى: {mode_lbl}", "SYSTEM")
                elif cmd == "PANIC_CLOSE" and brain.open_position is not None:
                    brain._close_position(
                        brain._exit_quote(brain.open_position["side"]),
                        "إغلاق فوري بطلب المالك (Manual Emergency Exit)",
                    )
            await stream_manager.broadcast_state()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WS error:", repr(e))
    finally:
        if ws in stream_manager.clients:
            stream_manager.clients.remove(ws)


# ==============================================================================
# 5. واجهة React 18 الاحترافية (Apple Pro Dark Quantitative Workstation)
# ==============================================================================
HTML_UI = """<!DOCTYPE html>
<html lang="ar" dir="rtl" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>XAU/USD — Quantitative Cortex React Terminal</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-canvas: #06080D;
      --bg-card: rgba(13, 17, 26, 0.85);
      --border-glass: rgba(255, 255, 255, 0.07);
    }
    * { box-sizing: border-box; }
    body {
      background-color: var(--bg-canvas);
      background-image:
        radial-gradient(at 15% 0%, rgba(245, 158, 11, 0.06) 0px, transparent 45%),
        radial-gradient(at 85% 5%, rgba(16, 185, 129, 0.05) 0px, transparent 45%);
      color: #F1F5F9;
      font-family: 'Cairo', sans-serif;
      margin: 0;
      overflow-x: hidden;
    }
    .mono {
      font-family: 'JetBrains Mono', monospace;
      font-variant-numeric: tabular-nums;
      direction: ltr;
      display: inline-block;
    }
    .glass-card {
      background: var(--bg-card);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--border-glass);
      border-radius: 16px;
      box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.55);
    }
    .glass-pill {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 10px;
    }
    @keyframes pulseGlow {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.45; transform: scale(0.92); }
    }
    .dot-live { animation: pulseGlow 2s infinite ease-in-out; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.2); border-radius: 9999px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.35); }
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel">
    const { useState, useEffect, useRef, useMemo } = React;

    // شارة فئة التفكير
    const categoryBadge = (cat) => {
      const map = {
        EXECUTION: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
        EXIT_DECISION: "bg-sky-500/15 text-sky-300 border-sky-500/30",
        RISK_SHIELD: "bg-rose-500/15 text-rose-300 border-rose-500/30",
        PERCEPTION: "bg-amber-500/15 text-amber-300 border-amber-500/30",
        SYSTEM: "bg-purple-500/15 text-purple-300 border-purple-500/30",
      };
      return map[cat] || "bg-slate-700/40 text-slate-300 border-slate-600/40";
    };

    // مكون شارت TradingView Lightweight المدمج مع React
    function TradingChart({ candles, latestCandle, zones, openPos, poc, vah, val, showZones, showVP }) {
      const containerRef = useRef(null);
      const chartRef = useRef(null);
      const seriesRef = useRef(null);
      const zoneLinesRef = useRef([]);
      const vpLinesRef = useRef([]);
      const posLinesRef = useRef([]);

      useEffect(() => {
        if (!containerRef.current) return;
        const chart = LightweightCharts.createChart(containerRef.current, {
          layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: '#94A3B8',
            fontFamily: 'JetBrains Mono',
            fontSize: 11,
          },
          grid: {
            vertLines: { color: 'rgba(255, 255, 255, 0.03)' },
            horzLines: { color: 'rgba(255, 255, 255, 0.03)' },
          },
          crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: { color: 'rgba(245, 158, 11, 0.35)', width: 1, style: 2 },
            horzLine: { color: 'rgba(245, 158, 11, 0.35)', width: 1, style: 2 },
          },
          rightPriceScale: {
            borderColor: 'rgba(255, 255, 255, 0.08)',
            autoScale: true,
          },
          timeScale: {
            borderColor: 'rgba(255, 255, 255, 0.08)',
            timeVisible: true,
            secondsVisible: false,
          },
        });

        const series = chart.addCandlestickSeries({
          upColor: '#10B981',
          downColor: '#F43F5E',
          borderUpColor: '#10B981',
          borderDownColor: '#F43F5E',
          wickUpColor: '#10B981',
          wickDownColor: '#F43F5E',
        });

        chartRef.current = chart;
        seriesRef.current = series;

        const ro = new ResizeObserver(entries => {
          if (!entries.length) return;
          const { width, height } = entries[0].contentRect;
          chart.applyOptions({ width, height });
        });
        ro.observe(containerRef.current);

        return () => {
          ro.disconnect();
          chart.remove();
        };
      }, []);

      // تحميل الشموع الأولية
      useEffect(() => {
        if (seriesRef.current && candles && candles.length > 0) {
          seriesRef.current.setData(candles);
          chartRef.current.timeScale().fitContent();
        }
      }, [candles]);

      // تحديث الشمعة الحية
      useEffect(() => {
        if (seriesRef.current && latestCandle) {
          seriesRef.current.update(latestCandle);
        }
      }, [latestCandle]);

      // رسم مناطق الجاذبية والشذوذ
      useEffect(() => {
        if (!seriesRef.current) return;
        zoneLinesRef.current.forEach(l => seriesRef.current.removePriceLine(l));
        zoneLinesRef.current = [];
        if (!showZones || !zones) return;

        zones.forEach(z => {
          const isRepulsor = z.type === 'REPULSOR_ANOMALY';
          const line = seriesRef.current.createPriceLine({
            price: z.price,
            color: isRepulsor ? 'rgba(245, 158, 11, 0.65)' : 'rgba(56, 189, 248, 0.50)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            axisLabelVisible: true,
            title: `${isRepulsor ? 'شذوذ رفض' : 'جاذبية'} ${z.strength}%`,
          });
          zoneLinesRef.current.push(line);
        });
      }, [zones, showZones]);

      // رسم خطوط Volume Profile (POC / VAH / VAL)
      useEffect(() => {
        if (!seriesRef.current) return;
        vpLinesRef.current.forEach(l => seriesRef.current.removePriceLine(l));
        vpLinesRef.current = [];
        if (!showVP) return;

        if (poc) {
          vpLinesRef.current.push(seriesRef.current.createPriceLine({
            price: poc,
            color: 'rgba(168, 85, 247, 0.85)',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            axisLabelVisible: true,
            title: 'VP POC',
          }));
        }
        if (vah) {
          vpLinesRef.current.push(seriesRef.current.createPriceLine({
            price: vah,
            color: 'rgba(168, 85, 247, 0.40)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: true,
            title: 'VAH',
          }));
        }
        if (val) {
          vpLinesRef.current.push(seriesRef.current.createPriceLine({
            price: val,
            color: 'rgba(168, 85, 247, 0.40)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: true,
            title: 'VAL',
          }));
        }
      }, [poc, vah, val, showVP]);

      // رسم خطوط الصفقة المفتوحة
      useEffect(() => {
        if (!seriesRef.current) return;
        posLinesRef.current.forEach(l => seriesRef.current.removePriceLine(l));
        posLinesRef.current = [];
        if (!openPos) return;

        posLinesRef.current.push(seriesRef.current.createPriceLine({
          price: openPos.entry_price,
          color: '#38BDF8',
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Solid,
          axisLabelVisible: true,
          title: `${openPos.side} (${openPos.lots}L)`,
        }));
        posLinesRef.current.push(seriesRef.current.createPriceLine({
          price: openPos.tp,
          color: '#10B981',
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Dotted,
          axisLabelVisible: true,
          title: 'STRUCTURAL TP',
        }));
        posLinesRef.current.push(seriesRef.current.createPriceLine({
          price: openPos.sl,
          color: '#F43F5E',
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Dotted,
          axisLabelVisible: true,
          title: openPos.be_secured ? 'SECURED SL (BE+)' : 'STRUCTURAL SL',
        }));
      }, [openPos]);

      return <div ref={containerRef} className="w-full h-full min-h-[410px]" />;
    }

    // التطبيق الرئيسي
    function App() {
      const [connected, setConnected] = useState(false);
      const [snapshotCandles, setSnapshotCandles] = useState([]);
      const [latestCandle, setLatestCandle] = useState(null);
      const [state, setState] = useState({
        tf_min: 1,
        candle_rem_s: 60,
        feed_source: "CONNECTING...",
        session: { name: "INIT", label: "جاري تحديد الجلسة...", bias: "NEUTRAL" },
        bot_active: true,
        lot_mode: "FIXED_001",
        price: null,
        bid: null,
        ask: null,
        spread: 0.25,
        balance: 10000,
        equity: 10000,
        kpis: { win_rate: 0, total_trades: 0, wins: 0, losses: 0, profit_factor: 0, net_pnl: 0, expectancy: 0, max_drawdown_pct: 0 },
        cognition: {
          hurst: 0.5, hurst_rank: 50, entropy: 0.5, entropy_rank: 50,
          velocity: 0, sigma: 0, atr: 1.2, z_score: 0,
          poc: null, vah: null, val: null, smc_event: "NONE",
          regime: "INITIALIZING", policy: "OBSERVING", confidence: 0,
          weights: {}, block_reason: "تجميع البيانات..."
        },
        discovered_zones: [],
        open_position: null,
        trade_log: [],
        thoughts: [],
      });

      const [priceFlash, setPriceFlash] = useState("text-white");
      const prevPriceRef = useRef(null);
      const wsRef = useRef(null);

      // إعدادات عرض الشارت والتبويبات
      const [showZones, setShowZones] = useState(true);
      const [showVP, setShowVP] = useState(true);
      const [activeTab, setActiveTab] = useState("POSITION"); // POSITION | HISTORY | STREAM

      useEffect(() => {
        let timer;
        function connect() {
          const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
          const ws = new WebSocket(`${proto}//${window.location.host}/ws`);
          wsRef.current = ws;

          ws.onopen = () => setConnected(true);
          ws.onmessage = (e) => {
            const data = JSON.parse(e.data);
            if (data.price !== undefined && data.price !== null && prevPriceRef.current !== null) {
              if (data.price > prevPriceRef.current) {
                setPriceFlash("text-emerald-400");
                setTimeout(() => setPriceFlash("text-white"), 350);
              } else if (data.price < prevPriceRef.current) {
                setPriceFlash("text-rose-400");
                setTimeout(() => setPriceFlash("text-white"), 350);
              }
            }
            if (data.price !== undefined) prevPriceRef.current = data.price;

            if (data.type === "INIT_SNAPSHOT") {
              if (data.candles) setSnapshotCandles(data.candles);
              setState(prev => ({ ...prev, ...data }));
            } else if (data.type === "TICK_UPDATE") {
              if (data.candle) setLatestCandle(data.candle);
              setState(prev => ({ ...prev, ...data }));
            }
          };
          ws.onclose = () => {
            setConnected(false);
            timer = setTimeout(connect, 2500);
          };
        }
        connect();
        return () => clearTimeout(timer);
      }, []);

      const sendCommand = (command) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ command }));
        }
      };

      const { cognition: cog, kpis, open_position: pos, session } = state;

      // حساب نسبة تقدم السعر بين الوقف والهدف للصفقة المفتوحة
      const posProgress = useMemo(() => {
        if (!pos || !state.price) return 50;
        const totalSpan = Math.abs(pos.tp - pos.sl);
        if (totalSpan <= 1e-4) return 50;
        const distFromSl = pos.side === "BUY" ? (state.price - pos.sl) : (pos.sl - state.price);
        return Math.min(100, Math.max(0, (distFromSl / totalSpan) * 100));
      }, [pos, state.price]);

      const formatCountdown = (sec) => {
        const m = Math.floor(sec / 60);
        const s = sec % 60;
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
      };

      return (
        <div className="min-h-screen flex flex-col p-3 lg:p-4 gap-3.5 max-w-[1920px] mx-auto select-none">

          {/* 1. الشريط القيادي العلوي (Apple Glass Command Header) */}
          <header className="glass-card px-4 py-3 flex flex-wrap items-center justify-between gap-4">
            {/* الهوية ومصدر البث */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3">
                <div className="relative flex items-center justify-center w-9 h-9 rounded-xl bg-amber-500/10 border border-amber-500/25">
                  <span className={`w-2.5 h-2.5 rounded-full ${connected ? 'bg-emerald-400 dot-live' : 'bg-rose-500'}`}></span>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-extrabold text-lg tracking-tight text-white mono">XAU/USD</span>
                    <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/25">
                      QUANT + SMC CORTEX v2
                    </span>
                    <span className="text-[11px] px-2 py-0.5 rounded-full bg-white/5 text-slate-300 border border-white/10">
                      {session?.label}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mono mt-0.5">{state.feed_source}</p>
                </div>
              </div>

              <div className="h-9 w-[1px] bg-white/10 hidden xl:block"></div>

              {/* السعر اللحظي و Bid/Ask */}
              <div className="flex items-center gap-3">
                <div>
                  <div className="text-[10px] text-slate-400">سعر الذهب الفوري (CFD)</div>
                  <div className={`text-2xl font-extrabold transition-colors duration-200 mono ${priceFlash}`}>
                    {state.price ? `$${Number(state.price).toFixed(2)}` : '--.--'}
                  </div>
                </div>
                <div className="glass-pill px-2.5 py-1 text-[11px] mono flex flex-col justify-center">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-slate-400">Spread:</span>
                    <span className="text-amber-300 font-semibold">${Number(state.spread).toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between gap-2 text-[10px] text-slate-400">
                    <span>B: {state.bid ? state.bid.toFixed(2) : '--'}</span>
                    <span>/</span>
                    <span>A: {state.ask ? state.ask.toFixed(2) : '--'}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* مؤشرات الأداء المالي (KPI Strip) */}
            <div className="flex flex-wrap items-center gap-2.5">
              <div className="glass-pill px-3 py-1.5 min-w-[115px]">
                <div className="text-[10px] text-slate-400">الرصيد (Balance)</div>
                <div className="text-sm font-bold text-slate-100 mono">
                  ${Number(state.balance).toLocaleString('en-US', { minimumFractionDigits: 2 })}
                </div>
              </div>
              <div className="glass-pill px-3 py-1.5 min-w-[115px]">
                <div className="text-[10px] text-slate-400">الملكية (Equity)</div>
                <div className={`text-sm font-bold mono ${state.equity >= state.balance ? 'text-emerald-400' : 'text-rose-400'}`}>
                  ${Number(state.equity).toLocaleString('en-US', { minimumFractionDigits: 2 })}
                </div>
              </div>
              <div className="glass-pill px-3 py-1.5 min-w-[105px]">
                <div className="text-[10px] text-slate-400">نسبة النجاح</div>
                <div className="text-sm font-bold text-amber-400 mono">
                  {kpis.win_rate}% <span className="text-[11px] text-slate-400">({kpis.wins}W/{kpis.losses}L)</span>
                </div>
              </div>
              <div className="glass-pill px-3 py-1.5 min-w-[95px]">
                <div className="text-[10px] text-slate-400">معامل الربح (PF)</div>
                <div className="text-sm font-bold text-sky-400 mono">
                  {kpis.profit_factor === null ? '∞' : Number(kpis.profit_factor).toFixed(2)}
                </div>
              </div>
              <div className="glass-pill px-3 py-1.5 min-w-[95px]">
                <div className="text-[10px] text-slate-400">أقصى سحب (DD)</div>
                <div className="text-sm font-bold text-rose-400 mono">
                  {Number(kpis.max_drawdown_pct).toFixed(2)}%
                </div>
              </div>
            </div>

            {/* أزرار التحكم السريع */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => sendCommand('TOGGLE_LOT_MODE')}
                className="px-3 py-2 rounded-xl text-xs font-semibold bg-white/5 hover:bg-white/10 text-slate-200 border border-white/10 transition flex items-center gap-1.5"
                title="تبديل نظام حساب اللوت"
              >
                <span className="w-2 h-2 rounded-full bg-sky-400"></span>
                <span>{state.lot_mode === 'FIXED_001' ? 'اللوت: 0.01 ثابت' : 'اللوت: مخاطرة ديناميكية'}</span>
              </button>

              <button
                onClick={() => sendCommand('TOGGLE_BOT')}
                className={`px-3.5 py-2 rounded-xl text-xs font-bold transition border ${
                  state.bot_active
                    ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/25'
                    : 'bg-slate-800/80 text-slate-300 border-slate-700 hover:bg-slate-700'
                }`}
              >
                {state.bot_active ? 'الوكيل المستقل: نشط' : 'الوكيل المستقل: متوقف'}
              </button>

              {pos && (
                <button
                  onClick={() => sendCommand('PANIC_CLOSE')}
                  className="px-3.5 py-2 rounded-xl text-xs font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40 hover:bg-rose-500/30 transition"
                >
                  إغلاق فوري طارئ
                </button>
              )}
            </div>
          </header>

          {/* 2. الشبكة الرئيسية (Bento Workspace Grid) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5 flex-1">

            {/* العمود الأيمن/الأوسط: الشارت التفاعلي + منصة إدارة الصفقات (8 أعمدة) */}
            <div className="lg:col-span-8 flex flex-col gap-3.5">

              {/* بطاقة الشارت */}
              <div className="glass-card p-3.5 flex-1 flex flex-col min-h-[450px]">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 mb-2 border-b border-white/[0.06]">
                  <div className="flex items-center gap-2.5">
                    <span className="text-xs font-bold text-slate-200">شارت الهيكل الكمي والسيولة (XAU/USD {state.tf_min}m)</span>
                    <span className="glass-pill px-2 py-0.5 text-[11px] mono text-amber-300">
                      إغلاق الشمعة: {formatCountdown(state.candle_rem_s || 0)}
                    </span>
                    {cog.smc_event !== "NONE" && (
                      <span className="px-2 py-0.5 rounded-md text-[10px] font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 mono">
                        {cog.smc_event}
                      </span>
                    )}
                  </div>

                  {/* أدوات إظهار/إخفاء طبقات الشارت */}
                  <div className="flex items-center gap-2 text-[11px]">
                    <button
                      onClick={() => setShowZones(!showZones)}
                      className={`px-2.5 py-1 rounded-lg border transition ${
                        showZones ? 'bg-amber-500/15 text-amber-300 border-amber-500/30' : 'bg-white/5 text-slate-400 border-white/5'
                      }`}
                    >
                      مناطق الشذوذ (KDE)
                    </button>
                    <button
                      onClick={() => setShowVP(!showVP)}
                      className={`px-2.5 py-1 rounded-lg border transition ${
                        showVP ? 'bg-purple-500/15 text-purple-300 border-purple-500/30' : 'bg-white/5 text-slate-400 border-white/5'
                      }`}
                    >
                      بروفيل الحجم (POC/VA)
                    </button>
                  </div>
                </div>

                <div className="flex-1 w-full rounded-xl overflow-hidden relative">
                  <TradingChart
                    candles={snapshotCandles}
                    latestCandle={latestCandle}
                    zones={state.discovered_zones}
                    openPos={pos}
                    poc={cog.poc}
                    vah={cog.vah}
                    val={cog.val}
                    showZones={showZones}
                    showVP={showVP}
                  />
                </div>
              </div>

              {/* منصة العمليات السفلية بنظام التبويبات (Tabbed Dock) */}
              <div className="glass-card p-4">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-3 mb-3 border-b border-white/[0.06]">
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => setActiveTab("POSITION")}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                        activeTab === "POSITION" ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      الصفقة النشطة {pos ? `(#${pos.id})` : ''}
                    </button>
                    <button
                      onClick={() => setActiveTab("HISTORY")}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                        activeTab === "HISTORY" ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      سجل الصفقات ({state.trade_log.length})
                    </button>
                  </div>

                  <span className={`text-[11px] px-2.5 py-0.5 rounded-full font-semibold ${
                    pos
                      ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                      : 'bg-white/5 text-slate-400 border border-white/10'
                  }`}>
                    {pos ? `صفقة ${pos.side} مفعّلة — يديرها الوكيل لحظياً` : 'لا توجد صفقة مفتوحة — رصد مستمر'}
                  </span>
                </div>

                {/* محتوى التبويب الأول: الصفقة النشطة */}
                {activeTab === "POSITION" && (
                  pos ? (
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2.5 text-xs">
                        <div className="glass-pill p-2.5">
                          <span className="text-[10px] text-slate-400 block">الاتجاه / العقد</span>
                          <span className={`font-extrabold mono text-sm ${pos.side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {pos.side} ({pos.lots}L)
                          </span>
                        </div>
                        <div className="glass-pill p-2.5">
                          <span className="text-[10px] text-slate-400 block">سعر الدخول</span>
                          <span className="font-bold mono text-sm text-sky-300">${pos.entry_price.toFixed(2)}</span>
                        </div>
                        <div className="glass-pill p-2.5">
                          <span className="text-[10px] text-slate-400 block">السعر الحالي</span>
                          <span className="font-bold mono text-sm text-white">${pos.current_price.toFixed(2)}</span>
                        </div>
                        <div className="glass-pill p-2.5">
                          <span className="text-[10px] text-slate-400 block">الوقف الهيكلي (SL)</span>
                          <span className="font-bold mono text-sm text-rose-400">${pos.sl.toFixed(2)}</span>
                        </div>
                        <div className="glass-pill p-2.5">
                          <span className="text-[10px] text-slate-400 block">الهدف الهيكلي (TP)</span>
                          <span className="font-bold mono text-sm text-emerald-400">${pos.tp.toFixed(2)}</span>
                        </div>
                        <div className="glass-pill p-2.5">
                          <span className="text-[10px] text-slate-400 block">العائد للمخاطرة (R)</span>
                          <span className={`font-bold mono text-sm ${(pos.r_multiple || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                            {(pos.r_multiple || 0) >= 0 ? '+' : ''}{Number(pos.r_multiple || 0).toFixed(2)}R
                          </span>
                        </div>
                        <div className="glass-pill p-2.5">
                          <span className="text-[10px] text-slate-400 block">صافي الربح اللحظي</span>
                          <span className={`font-extrabold mono text-sm ${pos.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                          </span>
                        </div>
                      </div>

                      {/* شريط التقدم البصري بين الوقف والهدف */}
                      <div className="glass-pill p-3">
                        <div className="flex items-center justify-between text-[11px] mb-1.5">
                          <span className="text-rose-400 mono font-semibold">SL: ${pos.sl.toFixed(2)}</span>
                          <span className="text-slate-300 text-[11px]">
                            {pos.be_secured ? '🛡️ الصفقة مؤمّنة (Break-Even+ مفعل)' : '⏳ قيد المراقبة الهيكلية الحية'}
                          </span>
                          <span className="text-emerald-400 mono font-semibold">TP: ${pos.tp.toFixed(2)}</span>
                        </div>
                        <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-rose-500 via-amber-400 to-emerald-400 transition-all duration-300"
                            style={{ width: `${posProgress}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="py-6 text-center text-xs text-slate-400">
                      لا توجد صفقة نشطة حالياً. يقوم الوكيل بفحص تدفق الأوامر، فجوات السعر العادلة (FVG)، وإنتروبيا شانون قبل التنفيذ.
                    </div>
                  )
                )}

                {/* محتوى التبويب الثاني: سجل الصفقات التاريخية */}
                {activeTab === "HISTORY" && (
                  <div className="overflow-x-auto max-h-56">
                    {state.trade_log.length === 0 ? (
                      <div className="py-6 text-center text-xs text-slate-400">لا توجد صفقات مغلقة في السجل بعد.</div>
                    ) : (
                      <table className="w-full text-right text-xs">
                        <thead className="text-[11px] text-slate-400 border-b border-white/10">
                          <tr>
                            <th class="pb-2">التوقيت</th>
                            <th class="pb-2">النوع</th>
                            <th class="pb-2">الدخول</th>
                            <th class="pb-2">الخروج</th>
                            <th class="pb-2">اللوت</th>
                            <th class="pb-2">الصافي</th>
                            <th class="pb-2">سبب الخروج المستقل</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/[0.05]">
                          {state.trade_log.map((t) => (
                            <tr key={t.id} className="hover:bg-white/[0.02]">
                              <td className="py-2 mono text-slate-400">{t.closed_at}</td>
                              <td className={`py-2 mono font-bold ${t.side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>{t.side}</td>
                              <td className="py-2 mono">${Number(t.entry_price).toFixed(2)}</td>
                              <td className="py-2 mono">${Number(t.exit_price).toFixed(2)}</td>
                              <td className="py-2 mono">{t.lots}</td>
                              <td className={`py-2 mono font-bold ${t.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {t.pnl >= 0 ? '+' : ''}${Number(t.pnl).toFixed(2)}
                              </td>
                              <td className="py-2 text-slate-300">{t.reason_exit}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* العمود الأيسر: القشرة الكمية (Quantitative Cortex) + مجرى التفكير الحي (4 أعمدة) */}
            <div className="lg:col-span-4 flex flex-col gap-3.5">

              {/* لوحة الوعي الرياضي */}
              <div className="glass-card p-4 flex flex-col gap-3.5">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xs font-extrabold text-amber-400 tracking-wide">القشرة الكمية والهيكلية (Cortex)</h2>
                    <p className="text-[11px] text-slate-400 mt-0.5">{cog.block_reason}</p>
                  </div>
                  <div className="glass-pill px-3 py-1 text-right">
                    <span className="text-[10px] text-slate-400 block">قناعة الدخول</span>
                    <span className="text-sm font-extrabold text-amber-300 mono">{cog.confidence}%</span>
                  </div>
                </div>

                {/* شريط القناعة */}
                <div className="w-full h-2 bg-slate-800/90 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-l from-emerald-400 via-amber-400 to-sky-400 transition-all duration-300"
                    style={{ width: `${Math.min(100, cog.confidence)}%` }}
                  />
                </div>

                {/* شبكة المؤشرات الكمية */}
                <div className="grid grid-cols-2 gap-2.5 text-xs">
                  <div className="glass-pill p-2.5">
                    <div className="text-[10px] text-slate-400">النظام المكتشف (Regime)</div>
                    <div className="font-bold text-slate-100 mt-0.5 mono text-[11px] truncate">{cog.regime}</div>
                  </div>
                  <div className="glass-pill p-2.5">
                    <div className="text-[10px] text-slate-400">السياسة النشطة (Policy)</div>
                    <div className="font-bold text-sky-400 mt-0.5 mono text-[11px] truncate">{cog.policy}</div>
                  </div>
                  <div className="glass-pill p-2.5">
                    <div className="flex items-center justify-between text-[10px] text-slate-400">
                      <span>معامل هيرست (Hurst)</span>
                      <span className="mono text-emerald-400">رتبة {cog.hurst_rank}%</span>
                    </div>
                    <div className="font-bold text-emerald-300 mt-0.5 mono text-sm">{Number(cog.hurst).toFixed(3)}</div>
                  </div>
                  <div className="glass-pill p-2.5">
                    <div className="flex items-center justify-between text-[10px] text-slate-400">
                      <span>إنتروبيا شانون</span>
                      <span className="mono text-amber-400">رتبة {cog.entropy_rank}%</span>
                    </div>
                    <div className="font-bold text-amber-300 mt-0.5 mono text-sm">{Number(cog.entropy).toFixed(3)}</div>
                  </div>
                  <div className="glass-pill p-2.5">
                    <div className="text-[10px] text-slate-400">المدى الحقيقي (ATR 14)</div>
                    <div className="font-bold text-purple-300 mt-0.5 mono text-sm">${Number(cog.atr || 0).toFixed(2)}</div>
                  </div>
                  <div className="glass-pill p-2.5">
                    <div className="text-[10px] text-slate-400">الانحراف المعياري (Z-Score)</div>
                    <div className="font-bold text-slate-200 mt-0.5 mono text-sm">{Number(cog.z_score || 0).toFixed(2)}σ</div>
                  </div>
                </div>

                {/* بروفيل الحجم التقديري (POC / VAH / VAL) */}
                <div className="glass-pill p-2.5 flex items-center justify-between text-[11px]">
                  <span className="text-slate-400 font-semibold">Volume Profile:</span>
                  <div className="flex items-center gap-3 mono">
                    <span className="text-slate-300">VAL: <b>${cog.val ? cog.val.toFixed(2) : '--'}</b></span>
                    <span className="text-purple-300">POC: <b>${cog.poc ? cog.poc.toFixed(2) : '--'}</b></span>
                    <span className="text-slate-300">VAH: <b>${cog.vah ? cog.vah.toFixed(2) : '--'}</b></span>
                  </div>
                </div>

                {/* المناطق المكتشفة ذاتياً */}
                <div>
                  <div className="text-[11px] text-slate-400 mb-1.5 font-semibold">مناطق الشذوذ والجاذبية السعرية (KDE):</div>
                  <div className="grid grid-cols-2 gap-1.5 text-xs">
                    {state.discovered_zones.map((z, idx) => (
                      <div key={idx} className="glass-pill px-2.5 py-1.5 flex items-center justify-between">
                        <span className={`mono font-bold ${z.type === 'REPULSOR_ANOMALY' ? 'text-amber-300' : 'text-sky-300'}`}>
                          ${z.price.toFixed(2)}
                        </span>
                        <span className="text-[10px] text-slate-400">
                          {z.type === 'REPULSOR_ANOMALY' ? 'رفض' : 'جاذبية'} ({z.strength}%)
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* أوزان التعلم الذاتي للسياسات */}
                <div className="pt-2 border-t border-white/[0.06]">
                  <div className="text-[11px] text-slate-400 mb-1.5 font-semibold">أوزان التعلم المعزز (Reinforcement Weights):</div>
                  <div className="grid grid-cols-3 gap-1.5 text-[11px] mono">
                    {Object.entries(cog.weights || {}).map(([k, v]) => (
                      <div key={k} className="glass-pill px-2 py-1 text-center">
                        <span className="text-slate-400 block text-[9px]">{k.split('_')[0]}</span>
                        <span className="text-amber-300 font-bold">{v}x</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* سجل التفكير والاستنتاج الحي */}
              <div className="glass-card p-4 flex-1 flex flex-col min-h-[290px]">
                <div className="flex items-center justify-between pb-2.5 mb-2.5 border-b border-white/[0.06]">
                  <h3 className="text-xs font-extrabold text-slate-200">مجرى التفكير والاستنتاج الحي (Neural Stream)</h3>
                  <span className="text-[10px] text-emerald-400 mono">LIVE COGNITION</span>
                </div>
                <div className="flex-1 overflow-y-auto space-y-2 pr-1 text-xs max-h-[340px]">
                  {state.thoughts.length === 0 ? (
                    <div className="text-slate-500 text-center py-8">بانتظار تدفق البيانات لتحليل السوق...</div>
                  ) : (
                    state.thoughts.map((th, i) => (
                      <div key={i} className="glass-pill p-2.5 transition hover:border-white/15">
                        <div className="flex items-center justify-between text-[10px] mb-1">
                          <span className={`px-2 py-0.5 rounded border font-semibold mono ${categoryBadge(th.category)}`}>
                            {th.category}
                          </span>
                          <span className="mono text-slate-400">{th.time}</span>
                        </div>
                        <p className="text-slate-200 leading-relaxed">{th.text}</p>
                      </div>
                    ))
                  )}
                </div>
              </div>

            </div>
          </div>
        </div>
      );
    }

    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(<App />);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_terminal():
    return HTMLResponse(content=HTML_UI)
