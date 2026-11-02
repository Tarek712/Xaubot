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

DEFAULT_TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DEFAULT_TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SQ_URL = "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD"
SQ_PROFILE = os.getenv("SQ_PROFILE", "standard").lower()
POLL_S = 2.0
STALE_QUOTE_S = 120
TF_MIN = int(os.getenv("BOT_TF_MIN", "1"))
_YF = {1: "1m", 5: "5m", 15: "15m"}
if TF_MIN not in _YF:
    raise SystemExit("BOT_TF_MIN must be one of 1, 5, 15")
YF_INTERVAL = _YF[TF_MIN]
TF_SEC = TF_MIN * 60
MIN_CANDLES = 50
MAX_SPREAD = 2.0
ASSUMED_SPREAD = 0.25
ENTRY_CONFIDENCE = 82.0
SIGNAL_CONFIRM = 2
COOLDOWN_AFTER_WIN_S = 120 * TF_MIN
COOLDOWN_AFTER_LOSS_S = 300 * TF_MIN
MAX_CONSECUTIVE_LOSSES = 4
LOSS_STREAK_PAUSE_S = 1800 * TF_MIN
OPPOSITE_EXIT_MIN_HOLD_S = 180 * TF_MIN
STAGNATION_EXIT_S = 900 * TF_MIN


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%m-%d %H:%M:%S")


def today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_active_session() -> Dict[str, str]:
    now_utc = datetime.now(timezone.utc)
    h = now_utc.hour + now_utc.minute / 60.0
    if 21.0 <= h < 22.1:
        return {"name": "DAILY_ROLLOVER", "label": "إغلاق/توقف السيولة اليومي (Rollover)", "bias": "HOLD"}
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
                closed_at TEXT,
                trade_date TEXT
            )
        """)
        c.execute("PRAGMA table_info(trade_history)")
        cols = [row[1] for row in c.fetchall()]
        if "trade_date" not in cols:
            c.execute("ALTER TABLE trade_history ADD COLUMN trade_date TEXT")

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
# 2. محرك إشعارات وتقارير تلغرام غير المتزامن (Telegram Notification Engine)
# ==============================================================================
class TelegramNotifier:
    def __init__(self):
        self.bot_token: str = DEFAULT_TG_TOKEN
        self.chat_id: str = DEFAULT_TG_CHAT_ID
        self.enabled: bool = bool(self.bot_token and self.chat_id)
        self.last_report_date: str = ""
        self.last_status: str = "جاهز للربط" if not self.enabled else "مُفعّل وجاهز"
        self._client: Optional[httpx.AsyncClient] = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=8.0)
        return self._client

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()

    def is_configured(self) -> bool:
        return bool(self.enabled and self.bot_token.strip() and self.chat_id.strip())

    def masked_token(self) -> str:
        t = self.bot_token.strip()
        if len(t) <= 10:
            return t
        return f"{t[:6]}...{t[-4:]}"

    async def send_html(self, text: str, force: bool = False) -> bool:
        if not force and not self.is_configured():
            return False
        token = self.bot_token.strip()
        cid = self.chat_id.strip()
        if not token or not cid:
            self.last_status = "خطأ: التوكن أو معرف المحادثة فارغ"
            return False

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": cid,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            r = await self._http().post(url, json=payload)
            if r.status_code == 200:
                self.last_status = f"تم الإرسال بنجاح ({datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC)"
                return True
            err_desc = r.json().get("description", f"HTTP {r.status_code}")
            self.last_status = f"فشل الإرسال: {err_desc}"
            return False
        except Exception as e:
            self.last_status = f"خطأ اتصال بتلغرام: {repr(e)}"
            return False

    async def notify_entry(self, pos: Dict, session_label: str):
        if not self.is_configured():
            return
        icon = "🟢" if pos["side"] == "BUY" else "🔴"
        direction_ar = "شراء (BUY)" if pos["side"] == "BUY" else "بيع (SELL)"
        msg = (
            f"{icon} <b>دخول صفقة ذهب جديدة | XAU/USD</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📌 <b>العملية:</b> <code>{direction_ar}</code> (#{pos['id']})\n"
            f"📦 <b>حجم العقد:</b> <code>{pos['lots']} Lot</code>\n"
            f"🎯 <b>سعر الدخول:</b> <code>${pos['entry_price']:.2f}</code>\n"
            f"🛡️ <b>الوقف الهيكلي (SL):</b> <code>${pos['sl']:.2f}</code> (مخاطرة: ${pos['risk_usd']:.2f})\n"
            f"🏆 <b>الهدف الهيكلي (TP):</b> <code>${pos['tp']:.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🧠 <b>القناعة الكمية:</b> <code>{pos['confidence']}%</code>\n"
            f"⚙️ <b>السياسة:</b> <code>{pos['policy']}</code>\n"
            f"📊 <b>حالة السوق:</b> <code>{pos['regime']}</code>\n"
            f"🌍 <b>الجلسة:</b> {session_label}\n"
            f"🕒 <b>التوقيت:</b> <code>{pos['opened_at']} UTC</code>"
        )
        await self.send_html(msg)

    async def notify_secured(self, pos: Dict, note: str):
        if not self.is_configured():
            return
        msg = (
            f"🛡️ <b>تأمين صفقة نشطة | XAU/USD (#{pos['id']})</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📌 <b>النوع:</b> <code>{pos['side']} ({pos['lots']}L)</code>\n"
            f"🔒 <b>الوقف المحدث (SL):</b> <code>${pos['sl']:.2f}</code>\n"
            f"💵 <b>الربح المحجوز/الجزئي:</b> <code>${pos['realized']:+.2f}</code>\n"
            f"📝 <b>التفاصيل:</b> {note}"
        )
        await self.send_html(msg)

    async def notify_exit(self, rec: Dict, r_mult: float, balance: float, kpis: Dict):
        if not self.is_configured():
            return
        win = rec["pnl"] >= 0
        badge = "✅ <b>إغلاق صفقة رابحة</b>" if win else "❌ <b>إغلاق صفقة خاسرة</b>"
        pnl_str = f"+${rec['pnl']:.2f}" if win else f"-${abs(rec['pnl']):.2f}"
        r_str = f"{r_mult:+.2f}R"
        msg = (
            f"{badge} | <b>XAU/USD (#{rec['id']})</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📌 <b>الاتجاه والعقد:</b> <code>{rec['side']} ({rec['lots']} Lot)</code>\n"
            f"📥 <b>سعر الدخول:</b> <code>${rec['entry_price']:.2f}</code>\n"
            f"📤 <b>سعر الخروج:</b> <code>${rec['exit_price']:.2f}</code>\n"
            f"💰 <b>الصافي النهائي:</b> <code>{pnl_str} ({r_str})</code>\n"
            f"📝 <b>سبب الخروج:</b> {rec['reason_exit']}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🏦 <b>الرصيد المحدث:</b> <code>${balance:,.2f}</code>\n"
            f"📈 <b>نسبة النجاح الكلية:</b> <code>{kpis['win_rate']}% ({kpis['wins']}W/{kpis['losses']}L)</code>\n"
            f"🕒 <b>وقت الإغلاق:</b> <code>{rec['closed_at']} UTC</code>"
        )
        await self.send_html(msg)

    async def send_daily_report(self, report: Dict, force: bool = False) -> bool:
        if not force and not self.is_configured():
            return False
        net = report["daily_pnl"]
        net_icon = "🟢" if net >= 0 else "🔴"
        net_str = f"+${net:.2f}" if net >= 0 else f"-${abs(net):.2f}"
        weights_txt = " | ".join([f"{k.split('_')[0]}: {v}x" for k, v in report["weights"].items()])

        msg = (
            f"📊 <b>التقرير اليومي لإغلاق سوق الذهب | XAU/USD</b>\n"
            f"📅 <b>التاريخ:</b> <code>{report['date']}</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🔢 <b>صفقات اليوم:</b> <code>{report['daily_total']}</code> "
            f"(رابحة: <code>{report['daily_wins']}</code> | خاسرة: <code>{report['daily_losses']}</code>)\n"
            f"🎯 <b>نسبة نجاح اليوم:</b> <code>{report['daily_win_rate']}%</code>\n"
            f"{net_icon} <b>صافي ربح/خسارة اليوم:</b> <code>{net_str} ({report['daily_return_pct']:+.2f}%)</code>\n"
            f"🏆 <b>أفضل صفقة اليوم:</b> <code>+${report['best_trade']:.2f}</code>\n"
            f"⚠️ <b>أسوأ صفقة اليوم:</b> <code>${report['worst_trade']:.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🏦 <b>إجمالي الرصيد الحالي:</b> <code>${report['balance']:,.2f}</code>\n"
            f"📈 <b>إجمالي الصافي الكلي:</b> <code>${report['total_net_pnl']:+,.2f}</code>\n"
            f"🛡️ <b>أقصى سحب مسجل (Max DD):</b> <code>{report['max_dd_pct']:.2f}%</code>\n"
            f"⚖️ <b>أوزان التعلم الذاتي:</b>\n<code>{weights_txt}</code>"
        )
        return await self.send_html(msg, force=force)


# ==============================================================================
# 3. عقل الوكيل الذاتي المستقل (Autonomous Quantitative & SMC Cortex)
# ==============================================================================
class AutonomousQuantitativeBrain:
    POLICIES = (
        "MOMENTUM_CONTINUATION",
        "GRAVITY_MEAN_REVERSION",
        "VOLATILITY_EXPANSION",
    )

    def __init__(self, tg: TelegramNotifier):
        self.lock = asyncio.Lock()
        self.tg = tg
        self.bot_active: bool = True
        self.can_trade: bool = False
        self.lot_mode: str = "FIXED_001"
        self.balance: float = 100.0
        self.equity: float = 100.0
        self.initial_balance: float = 100.0
        self.peak_balance: float = 100.0
        self.max_drawdown_pct: float = 0.0

        self.candles: deque = deque(maxlen=400)
        self.last_price: Optional[float] = None
        self.bid_price: Optional[float] = None
        self.ask_price: Optional[float] = None
        self.spread: float = ASSUMED_SPREAD

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

                c.execute("SELECT key, value FROM runtime_state")
                for k, v in c.fetchall():
                    if not v:
                        continue
                    parsed = json.loads(v)
                    if k == "open_position" and isinstance(parsed, dict) and "side" in parsed:
                        self.open_position = parsed
                    elif k == "lot_mode" and parsed in ("FIXED_001", "DYNAMIC_RISK"):
                        self.lot_mode = parsed
                    elif k == "cooldown_until":
                        self.cooldown_until = float(parsed)
                    elif k == "telegram_config" and isinstance(parsed, dict):
                        if parsed.get("bot_token"):
                            self.tg.bot_token = str(parsed["bot_token"]).strip()
                        if parsed.get("chat_id"):
                            self.tg.chat_id = str(parsed["chat_id"]).strip()
                        if "enabled" in parsed:
                            self.tg.enabled = bool(parsed["enabled"])
                        self.tg.last_status = "مُحمّل من الذاكرة الدائمة" if self.tg.is_configured() else "غير مُفعّل"
                    elif k == "last_report_date":
                        self.tg.last_report_date = str(parsed)
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

    def update_telegram_config(self, bot_token: str, chat_id: str, enabled: bool):
        if bot_token.strip() and "..." not in bot_token:
            self.tg.bot_token = bot_token.strip()
        self.tg.chat_id = chat_id.strip()
        self.tg.enabled = bool(enabled and self.tg.bot_token and self.tg.chat_id)
        self._save_runtime_key("telegram_config", {
            "bot_token": self.tg.bot_token,
            "chat_id": self.tg.chat_id,
            "enabled": self.tg.enabled,
        })
        self.tg.last_status = "تم حفظ الإعدادات" if self.tg.is_configured() else "تم الإيقاف أو بيانات ناقصة"

    def build_daily_report_data(self, date_str: Optional[str] = None) -> Dict:
        target_date = date_str or today_utc_str()
        month_day_prefix = target_date[5:]
        conn = sqlite3.connect(DB_PATH)
        try:
            c = conn.cursor()
            c.execute(
                """
                SELECT pnl FROM trade_history
                WHERE trade_date = ? OR (trade_date IS NULL AND closed_at LIKE ?)
                """,
                (target_date, f"{month_day_prefix}%"),
            )
            rows = [float(r[0]) for r in c.fetchall()]
        finally:
            conn.close()

        total = len(rows)
        wins = sum(1 for x in rows if x > 0)
        losses = sum(1 for x in rows if x < 0)
        daily_pnl = round(sum(rows), 2)
        win_rate = round((wins / total) * 100.0, 1) if total else 0.0
        start_day_bal = max(100.0, self.balance - daily_pnl)
        daily_return_pct = round((daily_pnl / start_day_bal) * 100.0, 2)
        best_trade = round(max(rows), 2) if rows else 0.0
        worst_trade = round(min(rows), 2) if rows else 0.0

        return {
            "date": target_date,
            "daily_total": total,
            "daily_wins": wins,
            "daily_losses": losses,
            "daily_win_rate": win_rate,
            "daily_pnl": daily_pnl,
            "daily_return_pct": daily_return_pct,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "balance": self.balance,
            "total_net_pnl": round(self.balance - self.initial_balance, 2),
            "max_dd_pct": self.max_drawdown_pct,
            "weights": dict(self.policy_weights),
        }

    def log_thought(self, msg: str, category: str = "COGNITION"):
        self.thoughts.appendleft({
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "category": category,
            "text": msg,
        })

    def compute_hurst_exponent(self, prices: np.ndarray) -> float:
        if len(prices) < 45:
            return 0.5
        log_p = np.log(np.maximum(prices, 1e-4))
        lags = range(2, min(25, len(log_p) // 2))
        tau, valid_lags = [], []
        for lag in lags:
            diff = log_p[lag:] - log_p[:-lag]
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
        if len(returns) < 20:
            return 0.75
        std_r = float(np.std(returns))
        if std_r <= 1e-8:
            return 0.30
        num_bins = 10
        counts, _ = np.histogram(returns, bins=num_bins, density=False)
        smoothed = counts.astype(float) + 0.05
        probs = smoothed / np.sum(smoothed)
        entropy = -np.sum(probs * np.log2(probs)) / np.log2(num_bins)
        return round(float(np.clip(entropy, 0.05, 0.99)), 3)

    def compute_true_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
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
            in_candle = (lows <= level + bandwidth * 0.5) & (highs >= level - bandwidth * 0.5)
            if np.any(in_candle):
                vol_profile[i] = float(np.sum(ranges[in_candle]))
            near_mask = dist < (bandwidth * 1.2)
            if np.any(near_mask):
                rejection_energy[i] = float(np.mean(wicks[near_mask]))

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
        result = {"bull_sweep": False, "bear_sweep": False, "bull_fvg": False, "bear_fvg": False, "label": "NONE"}
        if len(closes) < 25:
            return result

        o1, h1, l1, c1 = opens[-1], highs[-1], lows[-1], closes[-1]
        rng1 = max(h1 - l1, 1e-4)
        lower_wick = min(o1, c1) - l1
        upper_wick = h1 - max(o1, c1)

        prev_low = float(np.min(lows[-16:-1]))
        prev_high = float(np.max(highs[-16:-1]))

        if l1 < prev_low and c1 > prev_low and (lower_wick / rng1) >= 0.40:
            result["bull_sweep"] = True
            result["label"] = "BULL_LIQUIDITY_SWEEP"
        elif h1 > prev_high and c1 < prev_high and (upper_wick / rng1) >= 0.40:
            result["bear_sweep"] = True
            result["label"] = "BEAR_LIQUIDITY_SWEEP"

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
                if (direction == 1 and smc["bull_fvg"]) or (direction == -1 and smc["bear_fvg"]):
                    conviction += 5.5
            else:
                conviction = 56.0 + min(15.0, abs(short_drift) * 10.0)
        else:
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

    @staticmethod
    def _rank(history, x: float) -> float:
        arr = np.asarray(history, dtype=float)
        return float(np.mean(arr <= x)) if len(arr) else 0.5

    def _warm_history(self, closes: np.ndarray):
        self.hist_hurst.clear()
        self.hist_entropy.clear()
        start = max(91, len(closes) - 300)
        for j in range(start, len(closes)):
            self.hist_hurst.append(self.compute_hurst_exponent(closes[j - 90:j]))
            self.hist_entropy.append(self.compute_shannon_entropy(np.diff(closes[max(0, j - 61):j])))

    def _analyze(self, closed: List[Dict]):
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
            req_confirm = 1 if (d.get("smc_event", "NONE") != "NONE" and d["score"] >= 87.0) else SIGNAL_CONFIRM
            self._streak = self._streak + 1 if d["direction"] == self._last_dir else 1
            self._last_dir = d["direction"]
            d["confirmed"] = self._streak >= req_confirm
            if not d["confirmed"]:
                d["reason"] = f"إشارة أولية — بانتظار تأكيد الإغلاق ({self._streak}/{req_confirm} شموع)"
        else:
            self._streak, self._last_dir = 0, 0
            d["confirmed"] = False

    async def check_and_send_daily_rollover_report(self):
        if not self.tg.is_configured():
            return
        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        if now_utc.hour == 21 and now_utc.minute >= 1 and self.tg.last_report_date != today_str:
            self.tg.last_report_date = today_str
            self._save_runtime_key("last_report_date", today_str)
            report = self.build_daily_report_data(today_str)
            sent = await self.tg.send_daily_report(report)
            if sent:
                self.log_thought(f"تم إرسال التقرير اليومي لإغلاق السوق ({today_str}) إلى تلغرام.", "SYSTEM")

    async def process_market_state(self):
        async with self.lock:
            await self.check_and_send_daily_rollover_report()

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
                await self._manage_open_position(decision)
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
                await self._open_position(decision)
            else:
                self.block_reason = decision.get("reason", "لا إعداد")

    def _exit_quote(self, side: str) -> float:
        half = self.spread / 2.0
        p = self.last_price - half if side == "BUY" else self.last_price + half
        return round(p, 2)

    def _compute_structural_stops(self, side: str, entry: float) -> tuple[float, float, float]:
        closed = list(self.candles)[:-1]
        recent = closed[-12:] if len(closed) >= 12 else closed
        atr = max(0.6, self.atr_value)

        if side == "BUY":
            swing_low = min(c["low"] for c in recent) if recent else entry - atr * 2.0
            raw_sl = min(entry - atr * 1.6, swing_low - atr * 0.35)
            sl_d = float(np.clip(entry - raw_sl, atr * 1.4, atr * 3.2))
            tp_d = round(sl_d * 1.75, 2)
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

    async def _open_position(self, decision: Dict):
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
        await self.tg.notify_entry(self.open_position, get_active_session()["label"])

    async def _manage_open_position(self, decision: Dict):
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

        can_partial = pos["lots"] >= 0.02
        be_trigger_r = 0.85 if can_partial else 1.15

        if not pos["be_secured"] and r_multiple >= be_trigger_r:
            lock = round(max(self.spread + 0.05, pos["initial_risk"] * 0.18), 2)
            pos["sl"] = round(pos["entry_price"] + lock if side == "BUY" else pos["entry_price"] - lock, 2)
            pos["be_secured"] = True

            note_msg = f"تأمين Break-Even+ عند {r_multiple:.2f}R ونقل الوقف إلى {pos['sl']:.2f}$"
            if not pos["partial_taken"] and can_partial:
                closed_lots = round(max(0.01, round(pos["lots"] * 0.5, 2)), 2)
                if pos["lots"] - closed_lots >= 0.01:
                    chunk = round(diff * closed_lots * 100.0, 2)
                    self.balance = round(self.balance + chunk, 2)
                    pos["realized"] = round(pos["realized"] + chunk, 2)
                    pos["lots"] = round(pos["lots"] - closed_lots, 2)
                    pos["partial_taken"] = True
                    note_msg = f"جني جزئي ({closed_lots}L) بربح {chunk:+.2f}$ ونقل الوقف لـ {pos['sl']:.2f}$"

            self.log_thought(f"تأمين ذكي: {note_msg}", "RISK_SHIELD")
            self._save_runtime_key("open_position", pos)
            await self.tg.notify_secured(pos, note_msg)

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
            await self._close_position(q, "ضرب الهدف الإحصائي الهيكلي (Structural TP Hit)")
            return
        if hit_sl:
            reason = "وقف الأرباح المتحرك (Secured Trailing SL)" if pos["be_secured"] else "ضرب وقف الحماية الهيكلي (Structural SL)"
            await self._close_position(q, reason)
            return

        opposite = (side == "BUY" and decision["direction"] == -1) or (side == "SELL" and decision["direction"] == 1)
        if opposite and decision["score"] >= 82.0 and held >= OPPOSITE_EXIT_MIN_HOLD_S:
            await self._close_position(q, f"خروج استباقي: انعكاس هيكلي مضاد بقوة {decision['score']}%")
            return

        if held > STAGNATION_EXIT_S and abs(r_multiple) < 0.25 and self.entropy > 0.84:
            await self._close_position(q, f"إلغاء الفرضية ذاتياً: تباطؤ الزخم وارتفاع الإنتروبيا لـ {self.entropy}")

    async def _close_position(self, exit_price: float, reason_exit: str):
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

        r_result = round(total_pnl / max(pos["risk_usd"], 1e-6), 2)
        trade_date = today_utc_str()

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
            "trade_date": trade_date,
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
            f"إغلاق [{pos['side']}] عند {exit_price:.2f}$ | الصافي: {total_pnl:+.2f}$ ({r_result:+.2f}R) | السبب: {reason_exit}",
            "EXIT_DECISION",
        )
        await self.tg.notify_exit(record, r_result, self.balance, self.get_kpi_metrics())

    def _persist_trade_and_lesson(self, rec: Dict, policy_name: str):
        conn = sqlite3.connect(DB_PATH)
        try:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO trade_history (side, entry_price, exit_price, lots, pnl, confidence, regime,
                                           reason_entry, reason_exit, opened_at, closed_at, trade_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec["side"], rec["entry_price"], rec["exit_price"], rec["lots"], rec["pnl"],
                    rec["confidence"], rec["regime"], rec["reason_entry"], rec["reason_exit"],
                    rec["opened_at"], rec["closed_at"], rec.get("trade_date", today_utc_str()),
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


tg_notifier = TelegramNotifier()
brain = AutonomousQuantitativeBrain(tg_notifier)


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
            await self.brain.tg.aclose()

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
            "telegram": {
                "enabled": b.tg.enabled,
                "configured": b.tg.is_configured(),
                "masked_token": b.tg.masked_token(),
                "chat_id": b.tg.chat_id,
                "last_status": b.tg.last_status,
                "last_report_date": b.tg.last_report_date,
            },
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


app = FastAPI(title="Autonomous XAU/USD Quantitative React Terminal v3", lifespan=lifespan)


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
                    brain.log_thought(f"تم تبديل حجم العقد إلى: {mode_lbl}", "SYSTEM")

                elif cmd == "PANIC_CLOSE" and brain.open_position is not None:
                    await brain._close_position(
                        brain._exit_quote(brain.open_position["side"]),
                        "إغلاق فوري بطلب المالك (Manual Emergency Exit)",
                    )

                elif cmd == "SAVE_TELEGRAM":
                    token = str(data.get("bot_token", "")).strip()
                    chat_id = str(data.get("chat_id", "")).strip()
                    enabled = bool(data.get("enabled", True))
                    brain.update_telegram_config(token, chat_id, enabled)
                    brain.log_thought(f"تم تحديث إعدادات إشعارات تلغرام ({brain.tg.last_status}).", "SYSTEM")

                elif cmd == "TEST_TELEGRAM":
                    test_msg = (
                        "🔔 <b>اختبار اتصال ناجح | XAU/USD Quant Cortex</b>\n"
                        "━━━━━━━━━━━━━━━━━━\n"
                        f"✅ البوت متصل الآن بقناتك/حسابك بنجاح.\n"
                        f"💰 <b>الرصيد الحالي:</b> <code>${brain.balance:,.2f}</code>\n"
                        f"📊 <b>سعر الذهب الحي:</b> <code>${brain.last_price or 0.0:.2f}</code>\n"
                        f"🕒 <b>التوقيت:</b> <code>{stamp()} UTC</code>"
                    )
                    ok = await brain.tg.send_html(test_msg, force=True)
                    brain.log_thought(
                        "تم إرسال رسالة اختبار تلغرام بنجاح." if ok else f"فشل اختبار تلغرام: {brain.tg.last_status}",
                        "SYSTEM",
                    )

                elif cmd == "SEND_DAILY_REPORT_NOW":
                    report = brain.build_daily_report_data()
                    ok = await brain.tg.send_daily_report(report, force=True)
                    brain.log_thought(
                        "تم إرسال التقرير اليومي إلى تلغرام يدوياً." if ok else f"تعذر إرسال التقرير اليومي: {brain.tg.last_status}",
                        "SYSTEM",
                    )

            await stream_manager.broadcast_state()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WS error:", repr(e))
    finally:
        if ws in stream_manager.clients:
            stream_manager.clients.remove(ws)


HTML_UI = """<!DOCTYPE html>
<html lang="ar" dir="rtl" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover" />
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
      --bg-card: rgba(13, 17, 26, 0.88);
      --border-glass: rgba(255, 255, 255, 0.08);
    }
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    body {
      background-color: var(--bg-canvas);
      background-image:
        radial-gradient(at 15% 0%, rgba(245, 158, 11, 0.07) 0px, transparent 45%),
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
      box-shadow: 0 12px 34px -8px rgba(0, 0, 0, 0.55);
    }
    .glass-pill {
      background: rgba(255, 255, 255, 0.035);
      border: 1px solid rgba(255, 255, 255, 0.075);
      border-radius: 11px;
    }
    .no-scrollbar::-webkit-scrollbar { display: none; }
    .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
    @keyframes pulseGlow {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.45; transform: scale(0.92); }
    }
    .dot-live { animation: pulseGlow 2s infinite ease-in-out; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.22); border-radius: 9999px; }
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel">
    const { useState, useEffect, useRef, useMemo } = React;

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
          handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
          handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
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
          if (width > 0 && height > 0) {
            chart.applyOptions({ width, height });
          }
        });
        ro.observe(containerRef.current);

        return () => {
          ro.disconnect();
          chart.remove();
        };
      }, []);

      useEffect(() => {
        if (seriesRef.current && candles && candles.length > 0) {
          seriesRef.current.setData(candles);
          chartRef.current.timeScale().fitContent();
        }
      }, [candles]);

      useEffect(() => {
        if (seriesRef.current && latestCandle) {
          seriesRef.current.update(latestCandle);
        }
      }, [latestCandle]);

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
            title: `${isRepulsor ? 'رفض' : 'جاذبية'} ${z.strength}%`,
          });
          zoneLinesRef.current.push(line);
        });
      }, [zones, showZones]);

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
            title: 'POC',
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
          title: 'TP',
        }));
        posLinesRef.current.push(seriesRef.current.createPriceLine({
          price: openPos.sl,
          color: '#F43F5E',
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Dotted,
          axisLabelVisible: true,
          title: openPos.be_secured ? 'SL (BE+)' : 'SL',
        }));
      }, [openPos]);

      return <div ref={containerRef} className="w-full h-[270px] sm:h-[360px] lg:h-[430px]" />;
    }

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
        balance: 100,
        equity: 100,
        kpis: { win_rate: 0, total_trades: 0, wins: 0, losses: 0, profit_factor: 0, net_pnl: 0, expectancy: 0, max_drawdown_pct: 0 },
        telegram: { enabled: false, configured: false, masked_token: "", chat_id: "", last_status: "غير مرتبط", last_report_date: "" },
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

      const [showZones, setShowZones] = useState(true);
      const [showVP, setShowVP] = useState(true);
      const [desktopDockTab, setDesktopDockTab] = useState("POSITION");
      const [mobileNav, setMobileNav] = useState("CHART");

      const [tgTokenInput, setTgTokenInput] = useState("");
      const [tgChatInput, setTgChatInput] = useState("");
      const [tgEnabledInput, setTgEnabledInput] = useState(true);
      const tgLoadedRef = useRef(false);

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

            if (data.telegram && !tgLoadedRef.current) {
              setTgChatInput(data.telegram.chat_id || "");
              setTgEnabledInput(Boolean(data.telegram.enabled));
              tgLoadedRef.current = true;
            }

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

      const sendCommand = (command, extra = {}) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ command, ...extra }));
        }
      };

      const saveTelegramSettings = (e) => {
        e.preventDefault();
        sendCommand("SAVE_TELEGRAM", {
          bot_token: tgTokenInput,
          chat_id: tgChatInput,
          enabled: tgEnabledInput,
        });
      };

      const { cognition: cog, kpis, open_position: pos, session, telegram: tg } = state;

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

      const renderActivePosition = () => {
        if (!pos) {
          return (
            <div className="py-5 text-center text-xs text-slate-400">
              لا توجد صفقة نشطة حالياً — الوكيل يفحص هياكل السيولة (SMC) وإنتروبيا السوق.
            </div>
          );
        }
        return (
          <div className="space-y-3">
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-xs">
              <div className="glass-pill p-2.5">
                <span className="text-[10px] text-slate-400 block">الاتجاه / اللوت</span>
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
                <span className="text-[10px] text-slate-400 block">الوقف (SL)</span>
                <span className="font-bold mono text-sm text-rose-400">${pos.sl.toFixed(2)}</span>
              </div>
              <div className="glass-pill p-2.5">
                <span className="text-[10px] text-slate-400 block">الهدف (TP)</span>
                <span className="font-bold mono text-sm text-emerald-400">${pos.tp.toFixed(2)}</span>
              </div>
              <div className="glass-pill p-2.5">
                <span className="text-[10px] text-slate-400 block">العائد (R)</span>
                <span className={`font-bold mono text-sm ${(pos.r_multiple || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                  {(pos.r_multiple || 0) >= 0 ? '+' : ''}{Number(pos.r_multiple || 0).toFixed(2)}R
                </span>
              </div>
              <div className="glass-pill p-2.5 col-span-2 sm:col-span-1">
                <span className="text-[10px] text-slate-400 block">الربح اللحظي</span>
                <span className={`font-extrabold mono text-sm ${pos.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                </span>
              </div>
            </div>

            <div className="glass-pill p-3">
              <div className="flex items-center justify-between text-[11px] mb-1.5">
                <span className="text-rose-400 mono font-semibold">SL: ${pos.sl.toFixed(2)}</span>
                <span className="text-slate-300 text-[11px]">
                  {pos.be_secured ? '🛡️ مؤمّنة (Break-Even+)' : '⏳ تحت المراقبة الحية'}
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
        );
      };

      const renderTradeHistory = () => {
        if (state.trade_log.length === 0) {
          return <div className="py-6 text-center text-xs text-slate-400">لا توجد صفقات مغلقة في السجل بعد.</div>;
        }
        return (
          <div>
            <div className="space-y-2 md:hidden max-h-[380px] overflow-y-auto pr-1">
              {state.trade_log.map((t) => (
                <div key={t.id} className="glass-pill p-3 space-y-1.5 text-xs">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-0.5 rounded font-bold mono ${
                        t.side === 'BUY' ? 'bg-emerald-500/15 text-emerald-400' : 'bg-rose-500/15 text-rose-400'
                      }`}>
                        {t.side} ({t.lots}L)
                      </span>
                      <span className="mono text-[11px] text-slate-400">{t.closed_at}</span>
                    </div>
                    <span className={`font-extrabold mono text-sm ${t.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {t.pnl >= 0 ? '+' : ''}${Number(t.pnl).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-300 mono">
                    <span>دخول: ${Number(t.entry_price).toFixed(2)}</span>
                    <span>←</span>
                    <span>خروج: ${Number(t.exit_price).toFixed(2)}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed">{t.reason_exit}</p>
                </div>
              ))}
            </div>

            <div className="hidden md:block overflow-x-auto max-h-56">
              <table className="w-full text-right text-xs">
                <thead className="text-[11px] text-slate-400 border-b border-white/10">
                  <tr>
                    <th className="pb-2">التوقيت</th>
                    <th className="pb-2">النوع</th>
                    <th className="pb-2">الدخول</th>
                    <th className="pb-2">الخروج</th>
                    <th className="pb-2">اللوت</th>
                    <th className="pb-2">الصافي</th>
                    <th className="pb-2">سبب الخروج المستقل</th>
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
            </div>
          </div>
        );
      };

      const renderTelegramSettings = () => (
        <div className="space-y-3 text-xs">
          <div className="flex flex-wrap items-center justify-between gap-2 glass-pill p-3">
            <div>
              <div className="font-bold text-slate-200 flex items-center gap-2">
                <span>حالة الربط بتلغرام:</span>
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                  tg.configured ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30' : 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                }`}>
                  {tg.configured ? 'مُفعّل ومتصل' : 'بانتظار الإعداد'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">{tg.last_status}</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => sendCommand("TEST_TELEGRAM")}
                className="px-3 py-1.5 rounded-lg font-bold bg-sky-500/15 text-sky-300 border border-sky-500/30 hover:bg-sky-500/25 transition"
              >
                🔔 إرسال رسالة اختبار
              </button>
              <button
                type="button"
                onClick={() => sendCommand("SEND_DAILY_REPORT_NOW")}
                className="px-3 py-1.5 rounded-lg font-bold bg-amber-500/15 text-amber-300 border border-amber-500/30 hover:bg-amber-500/25 transition"
              >
                📊 إرسال التقرير اليومي الآن
              </button>
            </div>
          </div>

          <form onSubmit={saveTelegramSettings} className="grid grid-cols-1 sm:grid-cols-12 gap-2.5 items-end">
            <div className="sm:col-span-5">
              <label className="block text-[11px] text-slate-400 mb-1">
                توكن البوت (Bot Token) {tg.masked_token ? `[الحالي: ${tg.masked_token}]` : ''}:
              </label>
              <input
                type="password"
                value={tgTokenInput}
                onChange={(e) => setTgTokenInput(e.target.value)}
                placeholder={tg.masked_token ? "اتركه فارغاً للإبقاء على التوكن المحفوظ" : "123456789:ABCdef..."}
                className="w-full px-3 py-2 rounded-xl bg-slate-900/90 border border-white/10 text-slate-100 mono text-xs focus:outline-none focus:border-amber-400"
              />
            </div>
            <div className="sm:col-span-4">
              <label className="block text-[11px] text-slate-400 mb-1">
                معرف القناة أو الحساب (Chat ID / @channel):
              </label>
              <input
                type="text"
                value={tgChatInput}
                onChange={(e) => setTgChatInput(e.target.value)}
                placeholder="-1001234567890 أو @channel"
                className="w-full px-3 py-2 rounded-xl bg-slate-900/90 border border-white/10 text-slate-100 mono text-xs focus:outline-none focus:border-amber-400"
              />
            </div>
            <div className="sm:col-span-3 flex items-center gap-2">
              <label className="flex items-center gap-1.5 cursor-pointer text-slate-300 text-xs select-none">
                <input
                  type="checkbox"
                  checked={tgEnabledInput}
                  onChange={(e) => setTgEnabledInput(e.target.checked)}
                  className="rounded accent-emerald-500 w-4 h-4"
                />
                <span>تفعيل</span>
              </label>
              <button
                type="submit"
                className="flex-1 px-3 py-2 rounded-xl font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 transition"
              >
                حفظ الربط
              </button>
            </div>
          </form>

          <div className="text-[11px] text-slate-400 flex flex-wrap items-center justify-between gap-2 pt-1">
            <span>• يرسل البوت تلقائياً: إشعار الدخول (SL/TP) + إشعار التأمين + نتيجة الخروج النهائية.</span>
            <span>• التقرير اليومي يُرسل تلقائياً بعد إغلاق السوق عند الساعة <b className="mono text-amber-300">21:01 UTC</b>.</span>
          </div>
        </div>
      );

      const renderCortexPanel = () => (
        <div className="glass-card p-3.5 sm:p-4 flex flex-col gap-3.5">
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

          <div className="w-full h-2 bg-slate-800/90 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-l from-emerald-400 via-amber-400 to-sky-400 transition-all duration-300"
              style={{ width: `${Math.min(100, cog.confidence)}%` }}
            />
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="glass-pill p-2.5">
              <div className="text-[10px] text-slate-400">النظام المكتشف</div>
              <div className="font-bold text-slate-100 mt-0.5 mono text-[11px] truncate">{cog.regime}</div>
            </div>
            <div className="glass-pill p-2.5">
              <div className="text-[10px] text-slate-400">السياسة النشطة</div>
              <div className="font-bold text-sky-400 mt-0.5 mono text-[11px] truncate">{cog.policy}</div>
            </div>
            <div className="glass-pill p-2.5">
              <div className="flex items-center justify-between text-[10px] text-slate-400">
                <span>هيرست (Hurst)</span>
                <span className="mono text-emerald-400">{cog.hurst_rank}%</span>
              </div>
              <div className="font-bold text-emerald-300 mt-0.5 mono text-sm">{Number(cog.hurst).toFixed(3)}</div>
            </div>
            <div className="glass-pill p-2.5">
              <div className="flex items-center justify-between text-[10px] text-slate-400">
                <span>إنتروبيا شانون</span>
                <span className="mono text-amber-400">{cog.entropy_rank}%</span>
              </div>
              <div className="font-bold text-amber-300 mt-0.5 mono text-sm">{Number(cog.entropy).toFixed(3)}</div>
            </div>
            <div className="glass-pill p-2.5">
              <div className="text-[10px] text-slate-400">المدى الحقيقي (ATR)</div>
              <div className="font-bold text-purple-300 mt-0.5 mono text-sm">${Number(cog.atr || 0).toFixed(2)}</div>
            </div>
            <div className="glass-pill p-2.5">
              <div className="text-[10px] text-slate-400">الانحراف (Z-Score)</div>
              <div className="font-bold text-slate-200 mt-0.5 mono text-sm">{Number(cog.z_score || 0).toFixed(2)}σ</div>
            </div>
          </div>

          <div className="glass-pill p-2.5 flex flex-wrap items-center justify-between gap-2 text-[11px]">
            <span className="text-slate-400 font-semibold">Volume Profile:</span>
            <div className="flex items-center gap-2.5 mono">
              <span className="text-slate-300">VAL: <b>${cog.val ? cog.val.toFixed(2) : '--'}</b></span>
              <span className="text-purple-300">POC: <b>${cog.poc ? cog.poc.toFixed(2) : '--'}</b></span>
              <span className="text-slate-300">VAH: <b>${cog.vah ? cog.vah.toFixed(2) : '--'}</b></span>
            </div>
          </div>

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

          <div className="pt-2 border-t border-white/[0.06]">
            <div className="text-[11px] text-slate-400 mb-1.5 font-semibold">أوزان التعلم المعزز:</div>
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
      );

      const renderNeuralStream = () => (
        <div className="glass-card p-3.5 sm:p-4 flex-1 flex flex-col min-h-[260px]">
          <div className="flex items-center justify-between pb-2.5 mb-2.5 border-b border-white/[0.06]">
            <h3 className="text-xs font-extrabold text-slate-200">مجرى التفكير والاستنتاج الحي (Neural Stream)</h3>
            <span className="text-[10px] text-emerald-400 mono">LIVE</span>
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
      );

      return (
        <div className="min-h-screen flex flex-col p-2.5 sm:p-3.5 lg:p-4 pb-20 lg:pb-4 gap-3 max-w-[1920px] mx-auto select-none">
          <header className="glass-card p-3 flex flex-col gap-2.5">
            <div className="flex items-center justify-between gap-2 flex-wrap">
              <div className="flex items-center gap-2.5">
                <div className="relative flex items-center justify-center w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-amber-500/10 border border-amber-500/25 shrink-0">
                  <span className={`w-2.5 h-2.5 rounded-full ${connected ? 'bg-emerald-400 dot-live' : 'bg-rose-500'}`}></span>
                </div>
                <div>
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="font-extrabold text-base sm:text-lg tracking-tight text-white mono">XAU/USD</span>
                    <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/25">
                      SMC v3
                    </span>
                    <span className="hidden sm:inline-block text-[10px] px-2 py-0.5 rounded-full bg-white/5 text-slate-300 border border-white/10">
                      {session?.label}
                    </span>
                  </div>
                  <p className="text-[10px] text-slate-400 mono truncate max-w-[190px] sm:max-w-none">{state.feed_source}</p>
                </div>
              </div>

              <div className="flex items-center gap-2.5">
                <div className="text-left sm:text-right">
                  <div className={`text-xl sm:text-2xl font-extrabold transition-colors duration-200 mono ${priceFlash}`}>
                    {state.price ? `$${Number(state.price).toFixed(2)}` : '--.--'}
                  </div>
                  <div className="text-[10px] text-slate-400 mono">
                    Spd: <b className="text-amber-300">${Number(state.spread).toFixed(2)}</b>
                  </div>
                </div>

                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => sendCommand('TOGGLE_LOT_MODE')}
                    className="px-2.5 py-1.5 rounded-xl text-[11px] font-semibold bg-white/5 hover:bg-white/10 text-slate-200 border border-white/10 transition"
                  >
                    {state.lot_mode === 'FIXED_001' ? '0.01L' : 'Risk%'}
                  </button>
                  <button
                    onClick={() => sendCommand('TOGGLE_BOT')}
                    className={`px-2.5 sm:px-3 py-1.5 rounded-xl text-[11px] font-bold transition border ${
                      state.bot_active
                        ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                        : 'bg-slate-800/80 text-slate-300 border-slate-700'
                    }`}
                  >
                    {state.bot_active ? 'نشط' : 'متوقف'}
                  </button>
                  {pos && (
                    <button
                      onClick={() => sendCommand('PANIC_CLOSE')}
                      className="px-2.5 py-1.5 rounded-xl text-[11px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40"
                    >
                      إغلاق!
                    </button>
                  )}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2 overflow-x-auto no-scrollbar pt-1 border-t border-white/[0.05]">
              <div className="glass-pill px-2.5 py-1 shrink-0 min-w-[105px]">
                <div className="text-[9px] text-slate-400">الرصيد (Balance)</div>
                <div className="text-xs sm:text-sm font-bold text-slate-100 mono">
                  ${Number(state.balance).toLocaleString('en-US', { minimumFractionDigits: 2 })}
                </div>
              </div>
              <div className="glass-pill px-2.5 py-1 shrink-0 min-w-[105px]">
                <div className="text-[9px] text-slate-400">الملكية (Equity)</div>
                <div className={`text-xs sm:text-sm font-bold mono ${state.equity >= state.balance ? 'text-emerald-400' : 'text-rose-400'}`}>
                  ${Number(state.equity).toLocaleString('en-US', { minimumFractionDigits: 2 })}
                </div>
              </div>
              <div className="glass-pill px-2.5 py-1 shrink-0 min-w-[100px]">
                <div className="text-[9px] text-slate-400">نسبة النجاح</div>
                <div className="text-xs sm:text-sm font-bold text-amber-400 mono">
                  {kpis.win_rate}% <span className="text-[10px] text-slate-400">({kpis.wins}W/{kpis.losses}L)</span>
                </div>
              </div>
              <div className="glass-pill px-2.5 py-1 shrink-0 min-w-[85px]">
                <div className="text-[9px] text-slate-400">معامل الربح</div>
                <div className="text-xs sm:text-sm font-bold text-sky-400 mono">
                  {kpis.profit_factor === null ? '∞' : Number(kpis.profit_factor).toFixed(2)}
                </div>
              </div>
              <div className="glass-pill px-2.5 py-1 shrink-0 min-w-[85px]">
                <div className="text-[9px] text-slate-400">أقصى سحب</div>
                <div className="text-xs sm:text-sm font-bold text-rose-400 mono">
                  {Number(kpis.max_drawdown_pct).toFixed(2)}%
                </div>
              </div>
              <div className="glass-pill px-2.5 py-1 shrink-0 min-w-[105px]">
                <div className="text-[9px] text-slate-400">إشعارات تلغرام</div>
                <div className={`text-xs font-bold ${tg.configured ? 'text-emerald-400' : 'text-slate-400'}`}>
                  {tg.configured ? '✈️ متصل' : 'غير متصل'}
                </div>
              </div>
            </div>
          </header>

          <div className="hidden lg:grid lg:grid-cols-12 gap-3.5 flex-1">
            <div className="lg:col-span-8 flex flex-col gap-3.5">
              <div className="glass-card p-3.5 flex-1 flex flex-col">
                <div className="flex items-center justify-between gap-2 pb-2.5 mb-2 border-b border-white/[0.06]">
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

              <div className="glass-card p-4">
                <div className="flex items-center justify-between gap-2 pb-3 mb-3 border-b border-white/[0.06]">
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => setDesktopDockTab("POSITION")}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                        desktopDockTab === "POSITION" ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      الصفقة النشطة {pos ? `(#${pos.id})` : ''}
                    </button>
                    <button
                      onClick={() => setDesktopDockTab("HISTORY")}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                        desktopDockTab === "HISTORY" ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      سجل الصفقات ({state.trade_log.length})
                    </button>
                    <button
                      onClick={() => setDesktopDockTab("TELEGRAM")}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                        desktopDockTab === "TELEGRAM" ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      ✈️ إشعارات وتقارير تلغرام
                    </button>
                  </div>
                </div>

                {desktopDockTab === "POSITION" && renderActivePosition()}
                {desktopDockTab === "HISTORY" && renderTradeHistory()}
                {desktopDockTab === "TELEGRAM" && renderTelegramSettings()}
              </div>
            </div>

            <div className="lg:col-span-4 flex flex-col gap-3.5">
              {renderCortexPanel()}
              {renderNeuralStream()}
            </div>
          </div>

          <div className="flex lg:hidden flex-col gap-3 flex-1">
            {mobileNav === "CHART" && (
              <>
                <div className="glass-card p-3 flex flex-col">
                  <div className="flex items-center justify-between gap-1.5 pb-2 mb-1.5 border-b border-white/[0.06]">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-bold text-slate-200">الشارت الحي ({state.tf_min}m)</span>
                      <span className="glass-pill px-2 py-0.5 text-[10px] mono text-amber-300">
                        {formatCountdown(state.candle_rem_s || 0)}
                      </span>
                    </div>
                    <div className="flex items-center gap-1 text-[10px]">
                      <button
                        onClick={() => setShowZones(!showZones)}
                        className={`px-2 py-0.5 rounded border ${
                          showZones ? 'bg-amber-500/15 text-amber-300 border-amber-500/30' : 'bg-white/5 text-slate-400 border-white/5'
                        }`}
                      >
                        KDE
                      </button>
                      <button
                        onClick={() => setShowVP(!showVP)}
                        className={`px-2 py-0.5 rounded border ${
                          showVP ? 'bg-purple-500/15 text-purple-300 border-purple-500/30' : 'bg-white/5 text-slate-400 border-white/5'
                        }`}
                      >
                        POC
                      </button>
                    </div>
                  </div>
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

                <div className="glass-card p-3.5">
                  <div className="flex items-center justify-between mb-2 pb-2 border-b border-white/[0.06]">
                    <span className="text-xs font-bold text-amber-400">الصفقة النشطة وحالة القرار</span>
                    <span className="text-[11px] font-bold mono text-amber-300">{cog.confidence}% قناعة</span>
                  </div>
                  <p className="text-[11px] text-slate-300 mb-2.5">{cog.block_reason}</p>
                  {renderActivePosition()}
                </div>
              </>
            )}

            {mobileNav === "CORTEX" && renderCortexPanel()}

            {mobileNav === "LOGS" && (
              <>
                <div className="glass-card p-3.5">
                  <h3 className="text-xs font-extrabold text-slate-200 pb-2 mb-2.5 border-b border-white/[0.06]">
                    سجل الصفقات ({state.trade_log.length})
                  </h3>
                  {renderTradeHistory()}
                </div>
                {renderNeuralStream()}
              </>
            )}

            {mobileNav === "TELEGRAM" && (
              <div className="glass-card p-3.5">
                <h3 className="text-xs font-extrabold text-sky-400 pb-2 mb-3 border-b border-white/[0.06]">
                  ✈️ إعدادات ربط تلغرام والتقرير اليومي
                </h3>
                {renderTelegramSettings()}
              </div>
            )}
          </div>

          <nav className="lg:hidden fixed bottom-0 left-0 right-0 z-50 bg-[#090D15]/95 backdrop-blur-xl border-t border-white/10 px-2 py-1.5 flex items-center justify-around">
            <button
              onClick={() => setMobileNav("CHART")}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 rounded-xl text-[11px] font-bold transition ${
                mobileNav === "CHART" ? 'text-amber-400 bg-amber-500/10' : 'text-slate-400'
              }`}
            >
              <span>📈</span>
              <span>الشارت والصفقة</span>
            </button>
            <button
              onClick={() => setMobileNav("CORTEX")}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 rounded-xl text-[11px] font-bold transition ${
                mobileNav === "CORTEX" ? 'text-amber-400 bg-amber-500/10' : 'text-slate-400'
              }`}
            >
              <span>🧠</span>
              <span>الوعي الكمي</span>
            </button>
            <button
              onClick={() => setMobileNav("LOGS")}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 rounded-xl text-[11px] font-bold transition ${
                mobileNav === "LOGS" ? 'text-amber-400 bg-amber-500/10' : 'text-slate-400'
              }`}
            >
              <span>📜</span>
              <span>السجل والأفكار</span>
            </button>
            <button
              onClick={() => setMobileNav("TELEGRAM")}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 rounded-xl text-[11px] font-bold transition ${
                mobileNav === "TELEGRAM" ? 'text-sky-400 bg-sky-500/10' : 'text-slate-400'
              }`}
            >
              <span>✈️</span>
              <span>التلغرام</span>
            </button>
          </nav>
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
