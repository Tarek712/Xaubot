import asyncio
import hmac
import json
import math
import os
import sqlite3
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# إرضاء فحص ZeroGPU على Hugging Face (يطلب وجود دالة @spaces.GPU عند الإقلاع).
# الدالة فارغة ولا تُستدعى؛ والبوت لا يستخدم GPU أصلاً.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_stub():
        return 0
except Exception:
    pass

try:
    import pandas as pd
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    ML_DEPS_OK = True
except ImportError:
    pd = joblib = HistGradientBoostingClassifier = None
    ML_DEPS_OK = False

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:
    LGBMClassifier = None
    HAVE_LGBM = False

# ==============================================================================
# 0. الإعدادات الأساسية ومعايير صفقات النخبة A+ وفلاتر السيولة الصارمة
# ==============================================================================
def _writable_path(path: str) -> str:
    """على Hugging Face قد يكون مجلد التطبيق للقراءة فقط: ننتقل إلى /tmp تلقائياً."""
    d = os.path.dirname(os.path.abspath(path))
    return path if os.access(d, os.W_OK) else os.path.join("/tmp", os.path.basename(path))


DB_PATH = _writable_path(os.getenv("BOT_DB_PATH", "autonomous_memory.db"))
ML_MODEL_PATH = _writable_path(os.getenv("GOLD_MODEL_PATH", "gold_model.joblib"))

DEFAULT_TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DEFAULT_TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# MEXC Contract API Endpoints (عقد الذهب الدائم XAU_USDT - مجاني ومفتوح بالكامل بدون API Key)
MEXC_BASE_URL = "https://contract.mexc.com/api/v1/contract"
MEXC_KLINE_URL = f"{MEXC_BASE_URL}/kline/XAU_USDT"
MEXC_DEPTH_URL = f"{MEXC_BASE_URL}/depth/XAU_USDT"
MEXC_TICKER_URL = f"{MEXC_BASE_URL}/ticker?symbol=XAU_USDT"

# رمز الدخول للوحة التحكم (WebSocket). إن تُرك فارغاً تكون اللوحة مفتوحة — لا يُنصح بذلك خارج الجهاز المحلي.
DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN", "").strip()

POLL_S = 1.5
TF_MIN = int(os.getenv("BOT_TF_MIN", "1"))
_MEXC_INTERVALS = {1: "Min1", 5: "Min5", 15: "Min15", 30: "Min30", 60: "Min60"}
MEXC_INTERVAL = _MEXC_INTERVALS.get(TF_MIN, "Min1")
TF_SEC = TF_MIN * 60

# --- معايير القنص الهيكلي الصارمة (A+ High-Quality Setups & Strict Liquidity) ---
INITIAL_CAPITAL = 1000.0         # الرصيد الأساسي ($1,000)
MIN_CANDLES = 50                 # أقل عدد شموع قبل بدء التحليل
MAX_SPREAD = 0.99                # سقف السبريد المسموح للدخول للتجربة ($0.99)
ASSUMED_SPREAD = 0.25            # سبريد افتراضي
ENTRY_CONFIDENCE = 84.0          # عتبة القناعة الصارمة لصفقات النخبة A+ (84.0%)
MIN_ADAPTIVE_CONFIDENCE = 82.0   # الحد الأدنى التكيفي المسموح للقناعة
MAX_ADAPTIVE_CONFIDENCE = 90.0   # السقف التكيفي للقناعة
SIGNAL_CONFIRM = 1               # تأكيد الدخول

# فلاتر السيولة وعمق الطلبات والحجم (Strict Liquidity Gatekeepers)
MIN_DEPTH_TOTAL_VOL = 60.0       # الحد الأدنى لمجموع عقود دفتر الأوامر (Bids + Asks) لاعتبار السيولة نشطة
MIN_OBI_THRESHOLD = 0.22         # الحد الأدنى لانحياز عمق الطلبات الصارم (|OBI| >= 0.22)
MIN_VOLUME_RATIO = 1.00          # الحد الأدنى لنسبة فوليوم الشمعة مقارنة بمتوسط 20 شمعة (منع التداول في الركود)
MIN_ATR_VOLATILITY = 0.85        # الحد الأدنى للـ ATR لمنع التداول في النطاقات الميتة
MIN_ML_PROB_THRESHOLD = 0.58     # الحد الأدنى لاحتمالية نموذج التعلم الآلي لقبول الصفقة

MIN_SL_DISTANCE = 3.00           # حد أدنى للستوب الهيكلي ($3.00)
MAX_SL_DISTANCE = 8.00           # أقصى ستوب هيكلي مسموح ($8.00)
MIN_STRUCTURAL_RR = 2.50         # أقل عائد للمخاطرة مقبول لصفقات النخبة (1:2.5)
MAX_STRUCTURAL_RR = 3.50         # سقف العائد للمخاطرة المستهدف (1:3.5)
FALLBACK_RR_RATIO = 3.00         # عائد افتراضي قياسي (1:3)
DEFAULT_RISK_PCT = 0.0050        # المخاطرة الافتراضية في الوضع الديناميكي (0.50%)
MAX_LOT_CAP = 0.50               # سقف اللوت الأقصى

COOLDOWN_AFTER_WIN_S = 120 * TF_MIN
COOLDOWN_AFTER_LOSS_S = 240 * TF_MIN
MAX_CONSECUTIVE_LOSSES = 3
LOSS_STREAK_PAUSE_S = 1800 * TF_MIN
OPPOSITE_EXIT_MIN_HOLD_S = 180 * TF_MIN
STAGNATION_EXIT_S = 900 * TF_MIN
AUTO_RETRAIN_INTERVAL_S = 3600


_BG_TASKS: set = set()


def spawn(coro) -> "asyncio.Task":
    """create_task مع الاحتفاظ بمرجع للمهمة (لمنع جمعها من الذاكرة) وتسجيل أي استثناء."""
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)

    def _done(t: "asyncio.Task"):
        _BG_TASKS.discard(t)
        if not t.cancelled() and t.exception() is not None:
            print("Background task error:", repr(t.exception()))

    task.add_done_callback(_done)
    return task


def clean_json(o):
    """يحوّل NaN/Infinity إلى None وأنواع numpy إلى أنواع بايثون كي لا ينهار JSON.parse في المتصفح."""
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%m-%d %H:%M:%S")


def today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_active_session() -> Dict[str, str]:
    """
    مصفاة الجلسات الصارمة:
    - حظر التداول نهائياً في فترة الـ Rollover (20:30 - 22:10 UTC).
    - حظر التداول في الجلسة الآسيوية وأواخر نيويورك لضعف السيولة (20:00 - 07:00 UTC).
    - السماح بفتح الصفقات فقط في جلسة لندن، تداخل لندن ونيويورك، وذروة جلسة نيويورك (07:00 - 20:00 UTC).
    """
    now_utc = datetime.now(timezone.utc)
    h = now_utc.hour + now_utc.minute / 60.0

    # عطلة نهاية الأسبوع أو الإغلاق الأسبوعي
    if now_utc.weekday() == 5 or (now_utc.weekday() == 4 and h >= 20.5) or (now_utc.weekday() == 6 and h < 22.2):
        return {
            "name": "WEEKEND_CLOSED",
            "label": "إغلاق عطلة نهاية الأسبوع (حظر التداول)",
            "bias": "HOLD",
            "high_liquidity": False,
        }

    # حظر الـ Rollover اليومي
    if 20.0 <= h < 22.5:
        return {
            "name": "DEADZONE_ROLLOVER",
            "label": "حظر الـ Rollover وانعدام السيولة (20:00-22:30 UTC)",
            "bias": "HOLD",
            "high_liquidity": False,
        }

    # حظر الجلسة الآسيوية بسبب ضعف السيولة والتذبذب الكاذب
    if h >= 22.5 or h < 7.0:
        return {
            "name": "ASIAN_LOW_LIQUIDITY",
            "label": "الجلسة الآسيوية — سيولة ضعيفة (محظور فتح صفقات)",
            "bias": "HOLD",
            "high_liquidity": False,
        }

    # ذروة السيولة العالمية: تداخل لندن ونيويورك (13:00 - 17:00 UTC)
    if 13.0 <= h < 17.0:
        return {
            "name": "LDN_NY_OVERLAP",
            "label": "تداخل لندن ونيويورك (ذروة السيولة المؤسساتية 🔥)",
            "bias": "MOMENTUM",
            "high_liquidity": True,
        }

    # جلسة لندن النشطة (07:00 - 13:00 UTC)
    if 7.0 <= h < 13.0:
        return {
            "name": "LONDON_SESSION",
            "label": "جلسة لندن (سيولة عالية واندفاع سعري)",
            "bias": "MOMENTUM",
            "high_liquidity": True,
        }

    # جلسة نيويورك النشطة قبل تلاشي السيولة (17:00 - 20:00 UTC)
    return {
        "name": "NEWYORK_SESSION",
        "label": "جلسة نيويورك النشطة (سيولة مؤسساتية)",
        "bias": "MOMENTUM",
        "high_liquidity": True,
    }


# ==============================================================================
# 1. محرك قاعدة البيانات الدائمة (SQLite WAL Engine)
# ==============================================================================
def get_db_conn() -> sqlite3.Connection:
    first = not os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH, timeout=12.0)
    if first:
        try:
            os.chmod(DB_PATH, 0o600)  # القاعدة قد تحوي توكن تلغرام: قراءة للمالك فقط
        except OSError:
            pass
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db():
    conn = get_db_conn()
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
        c.execute("""
            CREATE TABLE IF NOT EXISTS market_candles (
                tf_min INTEGER,
                time INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL DEFAULT 0,
                PRIMARY KEY (tf_min, time)
            )
        """)
        conn.commit()
    finally:
        conn.close()


init_db()


# ==============================================================================
# 1.5 محرك التعلم الآلي الذاتي المستمر (Auto-ML Triple-Barrier Self-Trainer)
# ==============================================================================
MIN_ML_ROWS = 140


def ml_make_features(df):
    d = df.copy()
    c, h, l = d["close"], d["high"], d["low"]
    prev_c = c.shift(1)

    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    f = pd.DataFrame(index=d.index)
    for n in (1, 2, 3, 5, 10, 20, 50):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (10, 20, 50, 100):
        ema = c.ewm(span=n, adjust=False).mean()
        f[f"dist_ema_{n}"] = (c - ema) / d["atr"]
    f["ema_slope_20"] = c.ewm(span=20, adjust=False).mean().diff(5) / d["atr"]

    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - ma20) / (2 * sd20.replace(0, np.nan))
    f["atr_pct"] = d["atr"] / c
    f["atr_ratio"] = d["atr"] / d["atr"].rolling(80).mean()
    f["vol_20"] = c.pct_change().rolling(20).std()
    f["range_atr"] = (h - l) / d["atr"]
    f["body_atr"] = (c - d["open"]) / d["atr"]
    f["upper_wick"] = (h - np.maximum(c, d["open"])) / d["atr"]
    f["lower_wick"] = (np.minimum(c, d["open"]) - l) / d["atr"]
    f["hh_20"] = (c - h.rolling(20).max()) / d["atr"]
    f["ll_20"] = (c - l.rolling(20).min()) / d["atr"]

    hr = d["time"].dt.hour
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    dow = d["time"].dt.dayofweek
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 5), np.cos(2 * np.pi * dow / 5)

    return f.replace([np.inf, -np.inf], np.nan), d["atr"]


def ml_barrier_outcomes(df, atr, tp_mult=2.8, sl_mult=1.1, horizon=24, spread=0.25):
    n = len(df)
    close, high, low = df["close"].values, df["high"].values, df["low"].values
    a = np.maximum(atr.values, 1.2)
    tp_d = np.maximum(tp_mult * a, MIN_SL_DISTANCE * MIN_STRUCTURAL_RR)
    sl_d = np.clip(sl_mult * a, MIN_SL_DISTANCE, MAX_SL_DISTANCE)

    res = {}
    for side in ("long", "short"):
        pnl = np.full(n, np.nan)
        done = np.zeros(n, dtype=bool)
        for k in range(1, horizon + 1):
            hi = np.full(n, np.nan)
            lo = np.full(n, np.nan)
            hi[: n - k], lo[: n - k] = high[k:], low[k:]
            if side == "long":
                hit_sl = lo <= close - sl_d
                hit_tp = hi >= close + tp_d
            else:
                hit_sl = hi >= close + sl_d
                hit_tp = lo <= close - tp_d
            valid = ~np.isnan(hi) & ~done & ~np.isnan(a)
            m_sl = valid & hit_sl
            pnl[m_sl] = -sl_d[m_sl]
            done |= m_sl
            m_tp = valid & hit_tp & ~done
            pnl[m_tp] = tp_d[m_tp]
            done |= m_tp
        idx = np.arange(n)
        end = np.minimum(idx + horizon, n - 1)
        timeout = ~done & ~np.isnan(a) & (idx + horizon < n)
        diff = close[end] - close
        pnl[timeout] = (diff if side == "long" else -diff)[timeout]
        res[side] = pnl - spread
    return res["long"], res["short"]


def _build_classifier(n_samples: int):
    min_leaf = max(10, min(60, n_samples // 12))
    if HAVE_LGBM:
        return LGBMClassifier(
            n_estimators=160, learning_rate=0.035, num_leaves=12,
            min_child_samples=min_leaf, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=5.0, random_state=42, verbose=-1
        )
    return HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.04, max_leaf_nodes=12,
        min_samples_leaf=min_leaf, l2_regularization=5.0, random_state=42
    )


class AutoSelfLearningEngine:
    def __init__(self, model_path: str = ML_MODEL_PATH):
        self.model_path = model_path
        self.model_long = None
        self.model_short = None
        self.features: List[str] = []
        self.threshold: float = MIN_ML_PROB_THRESHOLD
        self.ready: bool = False
        self.last_train_ts: float = 0.0
        self.train_samples: int = 0
        self.status_text: str = "بانتظار التجميع للتدريب الذاتي..."
        self._load_if_exists()

    def _load_if_exists(self):
        if not ML_DEPS_OK or not os.path.exists(self.model_path):
            return
        try:
            m = joblib.load(self.model_path)
            saved_tf = m.get("timeframe")
            if saved_tf and saved_tf != f"M{TF_MIN}":
                self.status_text = f"نموذج محفوظ لإطار {saved_tf} مختلف عن M{TF_MIN} — إعادة تدريب"
                return
            self.model_long = m["model_long"]
            self.model_short = m["model_short"]
            self.features = m["features"]
            self.threshold = float(m.get("threshold", MIN_ML_PROB_THRESHOLD))
            self.train_samples = int(m.get("samples", 0))
            self.ready = True
            self.status_text = f"Auto-ML نشط ({self.train_samples} عينة)"
        except Exception as e:
            self.status_text = f"إعادة معايرة النموذج ({e})"

    def train_on_candles_sync(self, candles_list: List[Dict]) -> bool:
        if not ML_DEPS_OK or len(candles_list) < 200:
            return False
        try:
            df = pd.DataFrame(candles_list)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df[["time", "open", "high", "low", "close"]].astype(
                {"open": float, "high": float, "low": float, "close": float}
            ).sort_values("time").drop_duplicates("time").reset_index(drop=True)

            X, atr = ml_make_features(df)
            horizon = 20
            pnl_long, pnl_short = ml_barrier_outcomes(df, atr, tp_mult=2.8, sl_mult=1.1, horizon=horizon, spread=ASSUMED_SPREAD)
            y_long = (pnl_long > 0).astype(int)
            y_short = (pnl_short > 0).astype(int)

            ok = X.notna().all(axis=1).values & ~np.isnan(pnl_long) & ~np.isnan(pnl_short)
            X_clean = X[ok].reset_index(drop=True)
            y_l = y_long[ok]
            y_s = y_short[ok]

            if len(X_clean) < 100 or len(np.unique(y_l)) < 2 or len(np.unique(y_s)) < 2:
                return False

            ml = _build_classifier(len(X_clean))
            ms = _build_classifier(len(X_clean))
            ml.fit(X_clean, y_l)
            ms.fit(X_clean, y_s)

            self.model_long = ml
            self.model_short = ms
            self.features = list(X_clean.columns)
            self.train_samples = len(X_clean)
            self.ready = True
            self.last_train_ts = time.time()
            self.status_text = f"Auto-ML مُدرّب ذاتياً ({self.train_samples} عينة)"

            joblib.dump({
                "model_long": ml,
                "model_short": ms,
                "features": self.features,
                "threshold": self.threshold,
                "samples": self.train_samples,
                "timeframe": f"M{TF_MIN}",
            }, self.model_path)
            return True
        except Exception as e:
            print("Auto-ML train error:", repr(e))
            return False

    def predict_probs(self, closed_candles: List[Dict]) -> Optional[Dict[str, float]]:
        if not self.ready or not ML_DEPS_OK or len(closed_candles) < MIN_ML_ROWS:
            return None
        try:
            df = pd.DataFrame(closed_candles[-260:])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df[["time", "open", "high", "low", "close"]].astype(
                {"open": float, "high": float, "low": float, "close": float}
            ).reset_index(drop=True)
            feats, _ = ml_make_features(df)
            row = feats.iloc[[-1]][self.features]
            if row.isna().any(axis=None):
                return None
            pl = float(self.model_long.predict_proba(row)[0, 1])
            ps = float(self.model_short.predict_proba(row)[0, 1])
            return {"long": round(pl, 4), "short": round(ps, 4)}
        except Exception:
            return None


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
        direction_ar = "شراء نخبة A+ (BUY)" if pos["side"] == "BUY" else "بيع نخبة A+ (SELL)"
        ob_info = f"🏛️ <b>الأوردر بلوك:</b> <code>{pos.get('ob_tag', 'INSTITUTIONAL_OB')}</code>\n" if pos.get('ob_tag') else ""
        msg = (
            f"{icon} <b>دخول صفقة ذهب عالية الجودة A+ | XAU/USD (MEXC)</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📌 <b>العملية:</b> <code>{direction_ar}</code> (#{pos['id']})\n"
            f"📦 <b>حجم العقد:</b> <code>{pos['lots']:.2f} Lot</code>\n"
            f"🎯 <b>سعر الدخول:</b> <code>${pos['entry_price']:.2f}</code>\n"
            f"🛡️ <b>الوقف خلف الـ OB/الهيكل (SL):</b> <code>${pos['sl']:.2f}</code> (مخاطرة: ${pos['risk_usd']:.2f})\n"
            f"🏆 <b>الهدف المقابل (TP):</b> <code>${pos['tp']:.2f}</code> ({pos.get('target_rr', 3.0):.2f}R)\n"
            f"🏛️ <b>محطة الهدف:</b> <code>{pos.get('tp_anchor', 'STRUCTURAL_ZONE')}</code>\n"
            f"{ob_info}"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🧠 <b>القناعة:</b> <code>{pos['confidence']}%</code> | <b>السياسة:</b> <code>{pos['policy']}</code>\n"
            f"📊 <b>ميزان السيولة (OBI):</b> <code>{pos.get('obi_val', '0.00')}</code>\n"
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
            f"📌 <b>النوع:</b> <code>{pos['side']} ({pos['lots']:.2f}L)</code>\n"
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
            f"📌 <b>الاتجاه والعقد:</b> <code>{rec['side']} ({rec['lots']:.2f} Lot)</code>\n"
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
            f"📊 <b>التقرير اليومي لإغلاق سوق الذهب | XAU/USD (MEXC)</b>\n"
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
# 3. عقل الوكيل المؤسساتي وكاشف الـ Order Blocks وعمق الطلبات (SMC Pro A+ Sniper)
# ==============================================================================
class AutonomousQuantitativeBrain:
    POLICIES = (
        "ORDER_BLOCK_SNIPER",
        "GRAVITY_MEAN_REVERSION",
        "VOLATILITY_EXPANSION",
    )

    def __init__(self, tg: TelegramNotifier):
        self.lock = asyncio.Lock()
        self.tg = tg
        self.auto_ml = AutoSelfLearningEngine(ML_MODEL_PATH)

        self.bot_active: bool = True
        self.can_trade: bool = False

        self.lot_mode: str = "FIXED"              # "FIXED" أو "DYNAMIC_RISK"
        self.fixed_lot: float = 0.01              # حجم اللوت الثابت
        self.risk_pct: float = DEFAULT_RISK_PCT   # نسبة المخاطرة في الوضع الديناميكي

        self.balance: float = INITIAL_CAPITAL
        self.equity: float = INITIAL_CAPITAL
        self.initial_balance: float = INITIAL_CAPITAL
        self.peak_balance: float = INITIAL_CAPITAL
        self.max_drawdown_pct: float = 0.0

        self.candles: deque = deque(maxlen=1500)
        self.last_price: Optional[float] = None
        self.bid_price: Optional[float] = None
        self.ask_price: Optional[float] = None
        self.spread: float = ASSUMED_SPREAD

        # بيانات عمق الطلبات والسيولة اللحظية L2 Order Book Depth
        self.order_book_imbalance: float = 0.0    # OBI من -1.0 إلى +1.0
        self.top_bid_wall: Optional[Dict] = None  # أكبر جدار شراء
        self.top_ask_wall: Optional[Dict] = None  # أكبر جدار بيع
        self.order_book_bids_vol: float = 0.0
        self.order_book_asks_vol: float = 0.0
        self.volume_ratio: float = 1.0            # نسبة فوليوم الشمعة إلى متوسط 20 شمعة
        self.liquidity_ok: bool = False           # حالة السيولة اللحظية (الجلسة + العمق + الفوليوم + السبريد)

        # محرك الـ Order Blocks (OB Engine)
        self.order_blocks: List[Dict] = []        # قائمة الأوردر بلوك النشطة والمخففة
        self.active_ob_count: int = 0
        self.nearest_bull_ob: Optional[Dict] = None
        self.nearest_bear_ob: Optional[Dict] = None

        # المؤشرات الكمية والهيكلية
        self.hurst: float = 0.50
        self.entropy: float = 0.50
        self.velocity: float = 0.0
        self.volatility_sigma: float = 0.0
        self.atr_value: float = 1.50
        self.z_score: float = 0.0
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.ema_fast_series: List[Dict] = []
        self.ema_slow_series: List[Dict] = []
        self.macro_bias: str = "NEUTRAL"

        # القمم والقيعان الهيكلية
        self.swing_highs: List[float] = []
        self.swing_lows: List[float] = []
        self.nearest_swing_high: Optional[float] = None
        self.nearest_swing_low: Optional[float] = None
        self.range_position_pct: float = 50.0

        # فوليوم بروفايل حقيقي مبني على عقود MEXC
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
        self.active_regime: str = "INITIALIZING"
        self.active_policy: str = "OBSERVING"
        self.confidence: float = 0.0

        self.policy_weights: Dict[str, float] = {p: 1.0 for p in self.POLICIES}
        self.context_memory: Dict[str, Dict[str, float]] = {}
        self.sl_buffer_mult: float = 1.00
        self.adaptive_threshold: float = ENTRY_CONFIDENCE
        self.ml_probs: Optional[Dict[str, float]] = None
        self._training_lock: bool = False

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
        self.log_thought(
            f"منظومة قناص النخبة A+ جاهزة — حظر تام لأوقات السيولة الضعيفة وعتبة قناعة {self.adaptive_threshold}% ({self.auto_ml.status_text}).",
            "SYSTEM",
        )

    def _save_runtime_key(self, key: str, val_obj):
        try:
            conn = get_db_conn()
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

    def _update_drawdown(self, current_val: float):
        changed = False
        if current_val > self.peak_balance:
            self.peak_balance = round(current_val, 2)
            changed = True
        if self.peak_balance > 0:
            dd = round((self.peak_balance - current_val) / self.peak_balance * 100.0, 2)
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd
                changed = True
        if changed:
            self._save_runtime_key("drawdown_metrics", {
                "peak_balance": self.peak_balance,
                "max_drawdown_pct": self.max_drawdown_pct,
            })

    def save_candle_to_db(self, candle: Dict):
        try:
            conn = get_db_conn()
            try:
                c = conn.cursor()
                c.execute(
                    """
                    INSERT OR REPLACE INTO market_candles (tf_min, time, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        TF_MIN, candle["time"], candle["open"], candle["high"],
                        candle["low"], candle["close"], candle.get("volume", 0.0)
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            print("Candle save error:", repr(e))

    def save_bulk_candles_to_db(self, candles_list: List[Dict]):
        if not candles_list:
            return
        try:
            conn = get_db_conn()
            try:
                c = conn.cursor()
                rows = [
                    (TF_MIN, x["time"], x["open"], x["high"], x["low"], x["close"], x.get("volume", 0.0))
                    for x in candles_list
                ]
                c.executemany(
                    """
                    INSERT OR REPLACE INTO market_candles (tf_min, time, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                c.execute(
                    """
                    DELETE FROM market_candles
                    WHERE tf_min = ? AND time NOT IN (
                        SELECT time FROM market_candles WHERE tf_min = ? ORDER BY time DESC LIMIT 2000
                    )
                    """,
                    (TF_MIN, TF_MIN),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            print("Bulk candle save error:", repr(e))

    def _restore_state(self):
        try:
            conn = get_db_conn()
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
                    elif k == "lot_config" and isinstance(parsed, dict):
                        mode = str(parsed.get("lot_mode", "FIXED"))
                        self.lot_mode = "DYNAMIC_RISK" if mode == "DYNAMIC_RISK" else "FIXED"
                        self.fixed_lot = round(float(np.clip(float(parsed.get("fixed_lot", 0.01)), 0.01, MAX_LOT_CAP)), 2)
                        self.risk_pct = float(np.clip(float(parsed.get("risk_pct", DEFAULT_RISK_PCT)), 0.0020, 0.0300))
                    elif k == "cooldown_until":
                        self.cooldown_until = float(parsed)
                    elif k == "drawdown_metrics" and isinstance(parsed, dict):
                        self.peak_balance = max(self.peak_balance, float(parsed.get("peak_balance", self.peak_balance)))
                        self.max_drawdown_pct = max(0.0, float(parsed.get("max_drawdown_pct", 0.0)))
                    elif k == "self_learning_state" and isinstance(parsed, dict):
                        self.context_memory = parsed.get("context_memory", {})
                        self.sl_buffer_mult = float(np.clip(float(parsed.get("sl_buffer_mult", 1.0)), 0.90, 1.65))
                        self.adaptive_threshold = float(np.clip(
                            float(parsed.get("adaptive_threshold", ENTRY_CONFIDENCE)),
                            MIN_ADAPTIVE_CONFIDENCE,
                            MAX_ADAPTIVE_CONFIDENCE,
                        ))
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

                c.execute(
                    "SELECT time, open, high, low, close, volume FROM ("
                    "SELECT time, open, high, low, close, volume FROM market_candles "
                    "WHERE tf_min = ? ORDER BY time DESC LIMIT 1500"
                    ") ORDER BY time ASC",
                    (TF_MIN,),
                )
                saved_candles = [
                    {
                        "time": int(r[0]), "open": float(r[1]), "high": float(r[2]),
                        "low": float(r[3]), "close": float(r[4]), "volume": float(r[5] or 0.0)
                    }
                    for r in c.fetchall()
                ]
                if self.open_position:
                    banked = float(self.open_position.get("realized", 0.0) or 0.0)
                    if banked:
                        self.balance = round(self.balance + banked, 2)
                        self.equity = self.balance
                        self.peak_balance = max(self.peak_balance, self.balance)

                if saved_candles:
                    self.candles.extend(saved_candles)
                    if len(saved_candles) >= 120:
                        closes_arr = np.array([x["close"] for x in saved_candles], dtype=float)
                        self._warm_history(closes_arr)
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

    def update_lot_settings(self, lot_mode: Optional[str] = None, fixed_lot: Optional[float] = None, risk_pct: Optional[float] = None):
        if lot_mode in ("FIXED", "DYNAMIC_RISK"):
            self.lot_mode = lot_mode
        if fixed_lot is not None and math.isfinite(float(fixed_lot)):
            self.fixed_lot = round(float(np.clip(fixed_lot, 0.01, MAX_LOT_CAP)), 2)
        if risk_pct is not None and math.isfinite(float(risk_pct)):
            self.risk_pct = float(np.clip(risk_pct, 0.0020, 0.0300))
        self._save_runtime_key("lot_config", {
            "lot_mode": self.lot_mode,
            "fixed_lot": self.fixed_lot,
            "risk_pct": self.risk_pct,
        })

    async def trigger_background_learning(self, force: bool = False):
        if self._training_lock:
            return
        now = time.time()
        if not force and self.auto_ml.ready and (now - self.auto_ml.last_train_ts < AUTO_RETRAIN_INTERVAL_S):
            return
        if len(self.candles) < 200:
            return

        self._training_lock = True
        try:
            candles_copy = list(self.candles)[:-1]
            trained = await asyncio.to_thread(self.auto_ml.train_on_candles_sync, candles_copy)
            if trained:
                self.log_thought(
                    f"اكتمل التدريب الذاتي (Auto-ML) على {self.auto_ml.train_samples} شمعة من فيوتشر MEXC.",
                    "SYSTEM",
                )
        finally:
            self._training_lock = False

    def _update_contextual_learning_on_close(self, pos: Dict, total_pnl: float, r_result: float, reason_exit: str):
        ctx_key = pos.get("context_key") or f"{get_active_session()['name']}|{pos.get('policy', 'DEFAULT')}"
        rec = self.context_memory.get(ctx_key, {"wins": 0, "total": 0, "r_sum": 0.0})
        rec["total"] = int(rec["total"]) + 1
        if total_pnl > 0:
            rec["wins"] = int(rec["wins"]) + 1
        rec["r_sum"] = round(float(rec["r_sum"]) + r_result, 3)
        self.context_memory[ctx_key] = rec

        if total_pnl < 0 and "SL" in reason_exit:
            self.sl_buffer_mult = round(float(np.clip(self.sl_buffer_mult + 0.08, 0.90, 1.65)), 2)
            self.adaptive_threshold = round(float(np.clip(
                self.adaptive_threshold + 0.9, MIN_ADAPTIVE_CONFIDENCE, MAX_ADAPTIVE_CONFIDENCE
            )), 1)
        elif total_pnl > 0:
            self.sl_buffer_mult = round(float(np.clip(self.sl_buffer_mult - 0.03, 0.92, 1.65)), 2)
            self.adaptive_threshold = round(float(np.clip(
                self.adaptive_threshold - 0.4, MIN_ADAPTIVE_CONFIDENCE, MAX_ADAPTIVE_CONFIDENCE
            )), 1)

        self._save_runtime_key("self_learning_state", {
            "context_memory": self.context_memory,
            "sl_buffer_mult": self.sl_buffer_mult,
            "adaptive_threshold": self.adaptive_threshold,
        })

    def update_telegram_config(self, bot_token: str, chat_id: str, enabled: bool):
        if bot_token.strip() and "..." not in bot_token:
            self.tg.bot_token = bot_token.strip()
        self.tg.chat_id = chat_id.strip()
        self.tg.enabled = bool(enabled and self.tg.bot_token and self.tg.chat_id)
        self._save_runtime_key("telegram_config", {
            "bot_token": "" if (DEFAULT_TG_TOKEN and self.tg.bot_token == DEFAULT_TG_TOKEN) else self.tg.bot_token,
            "chat_id": self.tg.chat_id,
            "enabled": self.tg.enabled,
        })
        self.tg.last_status = "تم حفظ الإعدادات" if self.tg.is_configured() else "تم الإيقاف أو بيانات ناقصة"

    def build_daily_report_data(self, date_str: Optional[str] = None) -> Dict:
        target_date = date_str or today_utc_str()
        month_day_prefix = target_date[5:]
        conn = get_db_conn()
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
        start_day_bal = max(10.0, self.balance - daily_pnl)
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

    # --------------------------------------------------------------- التحليل الهيكلي والـ Order Blocks
    @staticmethod
    def compute_ema_array(closes: np.ndarray, period: int) -> np.ndarray:
        n = len(closes)
        if n == 0:
            return np.array([], dtype=float)
        out = np.zeros(n, dtype=float)
        alpha = 2.0 / (period + 1.0)
        seed_len = min(period, n)
        seed = float(np.mean(closes[:seed_len]))
        for i in range(seed_len):
            out[i] = float(np.mean(closes[: i + 1]))
        val = seed
        for i in range(seed_len, n):
            val = alpha * float(closes[i]) + (1.0 - alpha) * val
            out[i] = val
        return np.round(out, 2)

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
            return 1.5
        prev_closes = closes[:-1]
        cur_highs = highs[1:]
        cur_lows = lows[1:]
        tr = np.maximum(
            cur_highs - cur_lows,
            np.maximum(np.abs(cur_highs - prev_closes), np.abs(cur_lows - prev_closes)),
        )
        n = min(period, len(tr))
        atr = float(np.mean(tr[-n:]))
        if atr <= 0.15:
            atr = max(1.0, float(closes[-1]) * 0.0004)
        return round(atr, 3)

    def detect_structural_swings(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray):
        n = len(closes)
        if n < 30:
            return

        sh_list, sl_list = [], []
        start_idx = max(4, n - 200)
        for i in range(start_idx, n - 2):
            if highs[i] >= np.max(highs[i - 4: i + 3]):
                lvl = round(float(highs[i]), 2)
                if not sh_list or abs(lvl - sh_list[-1]) > self.atr_value * 0.50:
                    sh_list.append(lvl)
            if lows[i] <= np.min(lows[i - 4: i + 3]):
                lvl = round(float(lows[i]), 2)
                if not sl_list or abs(lvl - sl_list[-1]) > self.atr_value * 0.50:
                    sl_list.append(lvl)

        self.swing_highs = sh_list[-10:]
        self.swing_lows = sl_list[-10:]

        cur_p = float(closes[-1])
        lookback = min(120, n)
        recent_high = float(np.max(highs[-lookback:]))
        recent_low = float(np.min(lows[-lookback:]))

        above_swings = [h for h in self.swing_highs if h >= cur_p - self.atr_value * 0.8]
        below_swings = [l for l in self.swing_lows if l <= cur_p + self.atr_value * 0.8]

        self.nearest_swing_high = round(min(above_swings) if above_swings else recent_high, 2)
        self.nearest_swing_low = round(max(below_swings) if below_swings else recent_low, 2)

        span = max(1.0, recent_high - recent_low)
        self.range_position_pct = round(float(np.clip((cur_p - recent_low) / span * 100.0, 0.0, 100.0)), 1)

    def detect_institutional_order_blocks(self, closed_candles: List[Dict]):
        """
        محرك كشف وتتبع مناطق الـ Order Blocks المؤسساتية النقية (A+ Unmitigated SMC Engine):
        1. الاندفاع الكاسر للهيكل (Strong Displacement >= 1.25 ATR & BOS)
        2. تحديد آخر شمعة معاكسة وتحديد مداها (Top/Bottom)
        3. تتبع التخفيف اللحظي وحصر الصفقات على البلوكات العذراء غير الملموسة (UNMITIGATED)
        """
        if len(closed_candles) < 35:
            return

        atr = self.atr_value
        obs: List[Dict] = []
        n = len(closed_candles)

        for i in range(12, n - 2):
            c_now = closed_candles[i]
            c_prev = closed_candles[i - 1]
            c_fwd = closed_candles[i + 1]

            # 1. كشف Bullish Order Block (اندفاع صاعد قوي >= 1.25 ATR يكسر قمة الشمعتين السابقتين)
            bullish_displacement = (
                c_now["close"] < c_now["open"]
                and (c_fwd["close"] - c_fwd["open"]) >= atr * 1.25
                and c_fwd["close"] > max(c_now["high"], c_prev["high"])
            )
            if bullish_displacement:
                ob_top = max(c_now["open"], c_now["close"])
                ob_bottom = c_now["low"]
                ob_time = c_now["time"]
                obs.append({
                    "id": f"BULL_{ob_time}",
                    "type": "BULLISH_OB",
                    "top": round(ob_top, 2),
                    "bottom": round(ob_bottom, 2),
                    "created_at": ob_time,
                    "status": "UNMITIGATED",
                    "touches": 0,
                })

            # 2. كشف Bearish Order Block (اندفاع هابط قوي >= 1.25 ATR يكسر قاع الشمعتين السابقتين)
            bearish_displacement = (
                c_now["close"] > c_now["open"]
                and (c_fwd["open"] - c_fwd["close"]) >= atr * 1.25
                and c_fwd["close"] < min(c_now["low"], c_prev["low"])
            )
            if bearish_displacement:
                ob_top = c_now["high"]
                ob_bottom = min(c_now["open"], c_now["close"])
                ob_time = c_now["time"]
                obs.append({
                    "id": f"BEAR_{ob_time}",
                    "type": "BEARISH_OB",
                    "top": round(ob_top, 2),
                    "bottom": round(ob_bottom, 2),
                    "created_at": ob_time,
                    "status": "UNMITIGATED",
                    "touches": 0,
                })

        cur_p = self.last_price or closed_candles[-1]["close"]
        active_obs = []

        for ob in obs[-15:]:
            created_idx = next((idx for idx, c in enumerate(closed_candles) if c["time"] == ob["created_at"]), None)
            if created_idx is None:
                continue

            # نفحص الشموع التي تلت الاندفاع باستثناء الشمعة الأخيرة (التي قد تكون شمعة اختبار الدخول الحالية)
            subsequent_candles = closed_candles[created_idx + 2: -1]
            mitigated = False
            touches = 0

            for sub_c in subsequent_candles:
                if ob["type"] == "BULLISH_OB":
                    if sub_c["close"] < ob["bottom"]:
                        mitigated = True
                        break
                    if sub_c["low"] <= ob["top"] and sub_c["low"] >= ob["bottom"]:
                        touches += 1
                else:
                    if sub_c["close"] > ob["top"]:
                        mitigated = True
                        break
                    if sub_c["high"] >= ob["bottom"] and sub_c["high"] <= ob["top"]:
                        touches += 1

            # التأكد من أن الشمعة المغلقة الأخيرة لم تكسر البلوك
            last_c = closed_candles[-1]
            if ob["type"] == "BULLISH_OB" and last_c["close"] < ob["bottom"]:
                mitigated = True
            elif ob["type"] == "BEARISH_OB" and last_c["close"] > ob["top"]:
                mitigated = True

            if not mitigated:
                ob["touches"] = touches
                ob["status"] = "TESTED" if touches > 0 else "UNMITIGATED"
                active_obs.append(ob)

        self.order_blocks = active_obs[-6:]
        self.active_ob_count = len(self.order_blocks)

        # لصفقات النخبة عالية الجودة: نعطي الأولوية القصوى للبلوكات العذراء (UNMITIGATED)
        bulls = [
            ob for ob in self.order_blocks
            if ob["type"] == "BULLISH_OB" and ob["status"] == "UNMITIGATED" and ob["top"] <= cur_p + atr * 0.5
        ]
        bears = [
            ob for ob in self.order_blocks
            if ob["type"] == "BEARISH_OB" and ob["status"] == "UNMITIGATED" and ob["bottom"] >= cur_p - atr * 0.5
        ]

        self.nearest_bull_ob = sorted(bulls, key=lambda x: cur_p - x["top"])[0] if bulls else None
        self.nearest_bear_ob = sorted(bears, key=lambda x: x["bottom"] - cur_p)[0] if bears else None

    def discover_volume_profile_and_liquidity(self, closed_candles: List[Dict]):
        if len(closed_candles) < 40:
            return

        highs = np.array([c["high"] for c in closed_candles[-120:]], dtype=float)
        lows = np.array([c["low"] for c in closed_candles[-120:]], dtype=float)
        closes = np.array([c["close"] for c in closed_candles[-120:]], dtype=float)
        vols = np.array([c.get("volume", 1.0) for c in closed_candles[-120:]], dtype=float)

        # حساب نسبة السيولة/الحجم للشمعة الأخيرة مقارنة بمتوسط آخر 20 شمعة
        recent_vols = vols[-21:-1] if len(vols) >= 21 else vols[:-1]
        avg_vol_20 = float(np.mean(recent_vols)) if len(recent_vols) > 0 else 1.0
        cur_vol = float(vols[-1])
        self.volume_ratio = round(cur_vol / max(0.1, avg_vol_20), 2)

        min_p, max_p = float(np.min(lows)), float(np.max(highs))
        span = max_p - min_p
        if span <= 1e-4:
            return

        bins_count = 35
        grid = np.linspace(min_p, max_p, bins_count)
        profile_vol = np.zeros(bins_count, dtype=float)
        bin_width = span / bins_count

        for i in range(len(closes)):
            c_lo, c_hi, c_vol = lows[i], highs[i], max(0.1, vols[i])
            affected = (grid >= c_lo - bin_width) & (grid <= c_hi + bin_width)
            if np.any(affected):
                profile_vol[affected] += c_vol / np.sum(affected)

        poc_idx = int(np.argmax(profile_vol))
        self.poc_price = round(float(grid[poc_idx]), 2)
        total_v = float(np.sum(profile_vol)) + 1e-9

        accum = float(profile_vol[poc_idx])
        lo_i, hi_i = poc_idx, poc_idx
        while (accum / total_v) < 0.70 and (lo_i > 0 or hi_i < bins_count - 1):
            next_lo = profile_vol[lo_i - 1] if lo_i > 0 else -1.0
            next_hi = profile_vol[hi_i + 1] if hi_i < bins_count - 1 else -1.0
            if next_hi >= next_lo and hi_i < bins_count - 1:
                hi_i += 1
                accum += float(profile_vol[hi_i])
            elif lo_i > 0:
                lo_i -= 1
                accum += float(profile_vol[lo_i])
            else:
                break

        self.val_price = round(float(grid[lo_i]), 2)
        self.vah_price = round(float(grid[hi_i]), 2)

        top_indices = np.argsort(profile_vol)[::-1][:4]
        zones = []
        for idx in top_indices:
            zones.append({
                "price": round(float(grid[idx]), 2),
                "strength": round(float(profile_vol[idx]) / float(np.max(profile_vol)) * 100.0, 1),
                "type": "LIQUIDITY_CORE" if idx == poc_idx else "VOLUME_NODE",
            })
        self.discovered_zones = sorted(zones, key=lambda x: x["price"], reverse=True)

    def detect_early_structural_trigger(
        self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> Dict:
        res = {
            "bull_ob_retest": False,
            "bear_ob_retest": False,
            "bull_sweep": False,
            "bear_sweep": False,
            "trigger_low": float(np.min(lows[-6:])),
            "trigger_high": float(np.max(highs[-6:])),
            "ob_target": None,
            "label": "NONE",
        }
        if len(closes) < 28:
            return res

        o1, h1, l1, c1 = float(opens[-1]), float(highs[-1]), float(lows[-1]), float(closes[-1])
        rng1 = max(h1 - l1, 1e-4)
        lower_wick1 = min(o1, c1) - l1
        upper_wick1 = h1 - max(o1, c1)

        prev_low_22 = float(np.min(lows[-24:-1]))
        prev_high_22 = float(np.max(highs[-24:-1]))

        # 1. ارتداد عالي الجودة من داخل Bullish Order Block عذراء (UNMITIGATED) مع شمعة رفض واضحة
        if self.nearest_bull_ob and self.nearest_bull_ob.get("status") == "UNMITIGATED":
            ob_top = self.nearest_bull_ob["top"]
            ob_bot = self.nearest_bull_ob["bottom"]
            in_ob_zone = (l1 <= ob_top + self.atr_value * 0.15) and (c1 >= ob_bot)
            strong_bull_reaction = (c1 > o1 and (c1 - o1) >= rng1 * 0.35) or (lower_wick1 / rng1 >= 0.45 and c1 >= o1)
            if in_ob_zone and strong_bull_reaction:
                res["bull_ob_retest"] = True
                res["ob_target"] = self.nearest_bull_ob
                res["label"] = "A+_BULL_OB_UNMITIGATED"

        # 2. ارتداد عالي الجودة من داخل Bearish Order Block عذراء (UNMITIGATED) مع شمعة رفض واضحة
        if self.nearest_bear_ob and self.nearest_bear_ob.get("status") == "UNMITIGATED":
            ob_top = self.nearest_bear_ob["top"]
            ob_bot = self.nearest_bear_ob["bottom"]
            in_ob_zone = (h1 >= ob_bot - self.atr_value * 0.15) and (c1 <= ob_top)
            strong_bear_reaction = (c1 < o1 and (o1 - c1) >= rng1 * 0.35) or (upper_wick1 / rng1 >= 0.45 and c1 <= o1)
            if in_ob_zone and strong_bear_reaction:
                res["bear_ob_retest"] = True
                res["ob_target"] = self.nearest_bear_ob
                res["label"] = "A+_BEAR_OB_UNMITIGATED"

        # 3. سحب السيولة المؤسساتي الكاسح (Liquidity Sweep + Strong Reversal Body)
        if l1 < prev_low_22 and c1 > prev_low_22 and c1 > o1 and (lower_wick1 / rng1 >= 0.35):
            res["bull_sweep"] = True
            if res["label"] == "NONE":
                res["label"] = "A+_BULL_LIQUIDITY_SWEEP"
        elif h1 > prev_high_22 and c1 < prev_high_22 and c1 < o1 and (upper_wick1 / rng1 >= 0.35):
            res["bear_sweep"] = True
            if res["label"] == "NONE":
                res["label"] = "A+_BEAR_LIQUIDITY_SWEEP"

        self.smc_event = res["label"]
        return res

    def check_strict_liquidity_conditions(self) -> Tuple[bool, str]:
        """
        مصفاة السيولة الصارمة (Strict Liquidity Gatekeeper):
        تمنع فتح أي صفقة إذا اختل أي شرط من شروط السيولة الخمسة:
        1. الجلسة الزمنية (منع الجلسة الآسيوية، التبييت، وعطلة الأسبوع).
        2. السبريد اللحظي (يجب ألا يتجاوز 0.99$).
        3. عمق دفتر الأوامر الكلي في MEXC (مجموع عقود الطلبات والعروض >= MIN_DEPTH_TOTAL_VOL).
        4. حجم تداول الشمعة مقارنة بمتوسط 20 شمعة (Volume Ratio >= MIN_VOLUME_RATIO).
        5. مدى الحركة السعرية ATR (منع التداول في النطاقات الميتة).
        """
        session_info = get_active_session()
        if not session_info.get("high_liquidity", False) or session_info["bias"] == "HOLD":
            return False, f"حظر السيولة الزمنية: {session_info['label']}"

        if self.spread > MAX_SPREAD:
            return False, f"حظر السيولة: السبريد الحالي ({self.spread:.2f}$) يتجاوز حد النخبة ({MAX_SPREAD:.2f}$)"

        total_depth = self.order_book_bids_vol + self.order_book_asks_vol
        if total_depth < MIN_DEPTH_TOTAL_VOL:
            return False, f"ضعف عمق دفتر الأوامر ({total_depth:.1f} < {MIN_DEPTH_TOTAL_VOL} عقد) — سيولة ضحلة"

        if self.volume_ratio < MIN_VOLUME_RATIO:
            return False, f"سيولة التداول ضعيفة: حجم الشمعة ({self.volume_ratio:.2f}x) أقل من متوسط 20 شمعة"

        if self.atr_value < MIN_ATR_VOLATILITY:
            return False, f"ركود سعري (ATR={self.atr_value:.2f}$ < {MIN_ATR_VOLATILITY}$) — انتظار تدفق السيولة"

        return True, "السيولة المؤسساتية مكتملة ونشطة"

    def synthesize_optimal_policy(
        self, times: np.ndarray, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> Dict:
        returns = np.diff(closes)
        if len(returns) < 40:
            self.active_regime = "CALIBRATING"
            return {"policy": "CALIBRATING", "direction": 0, "score": 0.0, "reason": "معايرة: بيانات غير كافية"}

        sigma = float(np.std(returns)) + 1e-6
        self.volatility_sigma = round(sigma, 3)

        ema50_arr = self.compute_ema_array(closes, 50)
        ema150_arr = self.compute_ema_array(closes, 150)
        self.ema_fast = float(ema50_arr[-1])
        self.ema_slow = float(ema150_arr[-1])
        self.ema_fast_series = [{"time": int(t), "value": float(v)} for t, v in zip(times[-350:], ema50_arr[-350:])]
        self.ema_slow_series = [{"time": int(t), "value": float(v)} for t, v in zip(times[-350:], ema150_arr[-350:])]

        cur_close = float(closes[-1])
        ema_gap = self.ema_fast - self.ema_slow
        # اشتراط اتجاه كلي واضح وقوي
        if ema_gap > self.atr_value * 0.25 and cur_close >= self.ema_slow:
            self.macro_bias = "BULLISH"
        elif ema_gap < -self.atr_value * 0.25 and cur_close <= self.ema_slow:
            self.macro_bias = "BEARISH"
        else:
            self.macro_bias = "NEUTRAL"

        self.detect_structural_swings(highs, lows, closes)
        trig = self.detect_early_structural_trigger(opens, highs, lows, closes)

        z_score = float((cur_close - np.mean(closes[-40:])) / (np.std(closes[-40:]) + 1e-6))
        self.z_score = round(z_score, 2)

        # 1. الفحص الصارم للسيولة أولاً قبل أي تفكير في فتح صفقة
        liq_ok, liq_reason = self.check_strict_liquidity_conditions()
        self.liquidity_ok = liq_ok
        if not liq_ok:
            self.active_regime = "LOW_LIQUIDITY_SHIELD"
            return {
                "policy": "LIQUIDITY_FILTER_HOLD",
                "direction": 0,
                "score": 25.0,
                "reason": liq_reason,
            }

        # 2. فلتر الضجيج العشوائي الصارم (Shannon Entropy Filter)
        if (len(self.hist_entropy) >= 60 and self.entropy_rank > 0.80) or self.entropy > 0.82:
            self.active_regime = "HIGH_ENTROPY_NOISE"
            return {
                "policy": "NOISE_FILTER_HOLD",
                "direction": 0,
                "score": 35.0,
                "reason": f"ضجيج عشوائي مرتفع (إنتروبيا {self.entropy}) — انتظار هيكل سعري نقي",
            }

        session_info = get_active_session()
        direction = 0
        conviction = 50.0
        policy = "ORDER_BLOCK_SNIPER"
        reason = ""
        matched_ob = None

        # ------------------------------------------------------------------
        # معايير صفقات النخبة (A+ High-Quality Confluence):
        # يشترط:
        # 1) توافق الاتجاه العام (Macro Bias Alignment) أو سحب سيولة مع ارتداد OB عذراء
        # 2) انحياز قوي في دفتر الأوامر (|OBI| >= MIN_OBI_THRESHOLD) في نفس اتجاه الصفقة
        # 3) وجود Order Block عذراء (UNMITIGATED)
        # ------------------------------------------------------------------
        obi = self.order_book_imbalance

        # فحص إعداد الشراء عالي الجودة (A+ BUY Setup)
        if trig["bull_ob_retest"]:
            if self.macro_bias != "BULLISH":
                reason = "رفض شراء: ارتداد من Bull OB لكن الاتجاه الكلي ليس صاعداً (يشترط توافق الهيكل)"
                conviction = 62.0
            elif obi < MIN_OBI_THRESHOLD:
                reason = f"رفض شراء: ارتداد من Bull OB لكن سيولة الشراء في دفتر الأوامر غير كافية (OBI={obi:+.2f} < +{MIN_OBI_THRESHOLD})"
                conviction = 66.0
            else:
                direction = 1
                matched_ob = trig["ob_target"]
                base = 84.0
                depth_bonus = min(6.0, (obi - MIN_OBI_THRESHOLD) * 15.0)
                vol_bonus = min(3.5, max(0.0, (self.volume_ratio - 1.0) * 3.0))
                sweep_bonus = 2.5 if trig["bull_sweep"] else 0.0
                session_bonus = 2.0 if session_info["name"] == "LDN_NY_OVERLAP" else 0.5
                conviction = base + depth_bonus + vol_bonus + sweep_bonus + session_bonus
                reason = (
                    f"صفقة نخبة A+ شراء: Bull OB عذراء ({matched_ob['bottom']}$-{matched_ob['top']}$) "
                    f"+ اتجاه صاعد + OBI ({obi:+.2f}) + Vol ({self.volume_ratio:.1f}x)"
                )

        # فحص إعداد البيع عالي الجودة (A+ SELL Setup)
        elif trig["bear_ob_retest"]:
            if self.macro_bias != "BEARISH":
                reason = "رفض بيع: ارتداد من Bear OB لكن الاتجاه الكلي ليس هابطاً (يشترط توافق الهيكل)"
                conviction = 62.0
            elif obi > -MIN_OBI_THRESHOLD:
                reason = f"رفض بيع: ارتداد من Bear OB لكن سيولة البيع في دفتر الأوامر غير كافية (OBI={obi:+.2f} > -{MIN_OBI_THRESHOLD})"
                conviction = 66.0
            else:
                direction = -1
                matched_ob = trig["ob_target"]
                base = 84.0
                depth_bonus = min(6.0, (abs(obi) - MIN_OBI_THRESHOLD) * 15.0)
                vol_bonus = min(3.5, max(0.0, (self.volume_ratio - 1.0) * 3.0))
                sweep_bonus = 2.5 if trig["bear_sweep"] else 0.0
                session_bonus = 2.0 if session_info["name"] == "LDN_NY_OVERLAP" else 0.5
                conviction = base + depth_bonus + vol_bonus + sweep_bonus + session_bonus
                reason = (
                    f"صفقة نخبة A+ بيع: Bear OB عذراء ({matched_ob['bottom']}$-{matched_ob['top']}$) "
                    f"+ اتجاه هابط + OBI ({obi:+.2f}) + Vol ({self.volume_ratio:.1f}x)"
                )

        # فحص إعداد سحب السيولة المؤسساتي فائق الجودة (A+ Institutional Liquidity Sweep)
        elif trig["bull_sweep"] and self.macro_bias == "BULLISH" and obi >= (MIN_OBI_THRESHOLD + 0.08) and self.volume_ratio >= 1.25:
            direction = 1
            policy = "GRAVITY_MEAN_REVERSION"
            conviction = 84.5 + min(5.0, obi * 10.0)
            reason = f"صفقة نخبة A+ سحب سيولة شرائي (Sweep) مع اتجاه صاعد وعمق شرائي قوي (OBI={obi:+.2f}, Vol={self.volume_ratio:.1f}x)"

        elif trig["bear_sweep"] and self.macro_bias == "BEARISH" and obi <= -(MIN_OBI_THRESHOLD + 0.08) and self.volume_ratio >= 1.25:
            direction = -1
            policy = "GRAVITY_MEAN_REVERSION"
            conviction = 84.5 + min(5.0, abs(obi) * 10.0)
            reason = f"صفقة نخبة A+ سحب سيولة بيعي (Sweep) مع اتجاه هابط وعمق بيعي قوي (OBI={obi:+.2f}, Vol={self.volume_ratio:.1f}x)"

        else:
            if not reason:
                reason = (
                    f"مراقبة بصبر لاقتناص إعداد A+ مكتمل الأركان "
                    f"(Bias: {self.macro_bias} | OBI: {obi:+.2f} | Vol: {self.volume_ratio:.1f}x)"
                )
            conviction = 52.0

        ctx_key = f"{session_info['name']}|{policy}"

        # 3. الفلتر الحاسم لنموذج التعلم الآلي (Auto-ML Veto Gatekeeper)
        if direction != 0:
            w = self.policy_weights.get(policy, 1.0)
            conviction += (w - 1.0) * 8.0

            if self.ml_probs is not None:
                p_fav = self.ml_probs["long"] if direction == 1 else self.ml_probs["short"]
                if p_fav < MIN_ML_PROB_THRESHOLD:
                    # إلغاء الصفقة فوراً إذا لم يؤكدها نموذج الـ Auto-ML باحتمالية قوية
                    direction = 0
                    conviction = 68.0
                    reason = f"تم إلغاء الإشارة بواسطة Auto-ML: احتمالية النجاح ({p_fav*100:.1f}%) أقل من حد النخبة ({MIN_ML_PROB_THRESHOLD*100:.0f}%)"
                else:
                    conviction += min(5.0, (p_fav - MIN_ML_PROB_THRESHOLD) * 35.0)

        self.active_regime = f"A+ MODE | {self.macro_bias} | OBI {obi:+.2f} | VOL {self.volume_ratio:.1f}x"
        final = round(float(np.clip(conviction, 15.0, 98.0)), 1)

        return {
            "policy": policy,
            "direction": direction,
            "score": final,
            "reason": reason,
            "z_score": round(z_score, 2),
            "smc_event": self.smc_event,
            "matched_ob": matched_ob,
            "context_key": ctx_key,
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
        times = np.array([c["time"] for c in closed], dtype=int)
        opens = np.array([c["open"] for c in closed], dtype=float)
        closes = np.array([c["close"] for c in closed], dtype=float)
        highs = np.array([c["high"] for c in closed], dtype=float)
        lows = np.array([c["low"] for c in closed], dtype=float)
        returns = np.diff(closes)

        if len(self.hist_hurst) < 60 and len(closes) >= 120:
            self._warm_history(closes)

        self.atr_value = self.compute_true_atr(highs, lows, closes, period=14)
        self.hurst = self.compute_hurst_exponent(closes[-90:])
        self.entropy = self.compute_shannon_entropy(returns[-60:])
        self.hist_hurst.append(self.hurst)
        self.hist_entropy.append(self.entropy)
        self.hurst_rank = self._rank(self.hist_hurst, self.hurst)
        self.entropy_rank = self._rank(self.hist_entropy, self.entropy)
        self.velocity = round(float(np.mean(returns[-3:])), 3)

        self.detect_institutional_order_blocks(closed)
        self.discover_volume_profile_and_liquidity(closed)
        self.ml_probs = self.auto_ml.predict_probs(closed)

        self._decision = self.synthesize_optimal_policy(times, opens, highs, lows, closes)
        self.active_policy = self._decision["policy"]
        self.confidence = self._decision["score"]

        d = self._decision
        if d["direction"] != 0 and d["score"] >= self.adaptive_threshold:
            self._streak = self._streak + 1 if d["direction"] == self._last_dir else 1
            self._last_dir = d["direction"]
            d["confirmed"] = self._streak >= SIGNAL_CONFIRM
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
                self.block_reason = f"تجميع بيانات MEXC ({len(self.candles)}/{MIN_CANDLES} شمعة)"
                return

            forming_ts = self.candles[-1]["time"]
            if forming_ts != self._analysis_ts:
                self._analysis_ts = forming_ts
                closed_candles = list(self.candles)[:-1]
                if closed_candles:
                    self.save_candle_to_db(closed_candles[-1])
                self._analyze(closed_candles)
                spawn(self.trigger_background_learning(force=False))

            decision = self._decision

            if self.open_position is not None:
                self.block_reason = "صفقة نخبة A+ مفتوحة قيد الإدارة"
                await self._manage_open_position(decision)
                return

            self.equity = self.balance
            self._update_drawdown(self.equity)
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

            # إعادة التحقق اللحظي المباشر من شروط السيولة والسبريد قبل التنفيذ الفعلي
            liq_ok, liq_msg = self.check_strict_liquidity_conditions()
            self.liquidity_ok = liq_ok
            if not liq_ok:
                self.block_reason = liq_msg
                return

            if decision["direction"] != 0 and decision["score"] >= self.adaptive_threshold and decision.get("confirmed"):
                self.block_reason = "إشارة نخبة A+ مؤكدة في سيولة عالية — جاري التنفيذ"
                await self._open_position(decision)
            else:
                self.block_reason = decision.get("reason", "بانتظار صفقة عالية الجودة A+")

    def _exit_quote(self, side: str) -> float:
        half = self.spread / 2.0
        p = self.last_price - half if side == "BUY" else self.last_price + half
        return round(p, 2)

    def _compute_pure_market_structure_stops(
        self, side: str, entry: float, decision: Dict
    ) -> Optional[Tuple[float, float, float, str, float, Optional[str]]]:
        """
        حساب الـ SL والـ TP الصارم لصفقات النخبة:
        - يشترط أن لا يقل العائد للمخاطرة عن 1:2.5 (MIN_STRUCTURAL_RR) ولا توجد عوائق هيكلية قريبة تمنع وصول السعر.
        """
        atr = max(1.0, self.atr_value)
        buffer = round((self.spread + max(0.80, atr * 0.75)) * self.sl_buffer_mult, 2)
        matched_ob = decision.get("matched_ob")
        ob_tag = None

        if side == "BUY":
            if matched_ob and "bottom" in matched_ob:
                raw_sl = round(matched_ob["bottom"] - buffer, 2)
                ob_tag = f"BULL_OB ({matched_ob['bottom']}$-{matched_ob['top']}$)"
            elif self.nearest_swing_low:
                raw_sl = round(self.nearest_swing_low - buffer, 2)
            else:
                raw_sl = round(entry - atr * 2.2, 2)

            sl_d = round(entry - raw_sl, 2)
            if sl_d < MIN_SL_DISTANCE:
                sl_d = MIN_SL_DISTANCE
                raw_sl = round(entry - sl_d, 2)
            if sl_d > MAX_SL_DISTANCE:
                return None

            # منع فتح شراء إذا كان هناك أوردر بلوك بيعي قريب يعيق الوصول إلى 2.0R على الأقل
            if self.nearest_bear_ob and (self.nearest_bear_ob["bottom"] - entry) < sl_d * 2.0:
                return None

            candidates = []
            if self.poc_price and self.poc_price > entry:
                candidates.append((self.poc_price - 0.20, "VP_POC"))
            if self.nearest_bear_ob and self.nearest_bear_ob["bottom"] > entry:
                candidates.append((self.nearest_bear_ob["bottom"] - 0.20, "BEAR_OB_TARGET"))
            if self.top_ask_wall and self.top_ask_wall["price"] > entry:
                candidates.append((self.top_ask_wall["price"] - 0.15, "ASK_LIQUIDITY_WALL"))

            valid = [
                (tp, lbl, round((tp - entry) / sl_d, 2))
                for tp, lbl in candidates
                if (tp - entry) >= sl_d * MIN_STRUCTURAL_RR and (tp - entry) <= sl_d * MAX_STRUCTURAL_RR
            ]
            if valid:
                valid.sort(key=lambda x: x[0], reverse=True)
                tp_p, tp_lbl, rr = valid[0]
            else:
                rr = FALLBACK_RR_RATIO
                tp_p = round(entry + sl_d * rr, 2)
                tp_lbl = f"INSTITUTIONAL_TARGET ({rr}R)"

            return raw_sl, tp_p, sl_d, tp_lbl, rr, ob_tag

        else:
            if matched_ob and "top" in matched_ob:
                raw_sl = round(matched_ob["top"] + buffer, 2)
                ob_tag = f"BEAR_OB ({matched_ob['bottom']}$-{matched_ob['top']}$)"
            elif self.nearest_swing_high:
                raw_sl = round(self.nearest_swing_high + buffer, 2)
            else:
                raw_sl = round(entry + atr * 2.2, 2)

            sl_d = round(raw_sl - entry, 2)
            if sl_d < MIN_SL_DISTANCE:
                sl_d = MIN_SL_DISTANCE
                raw_sl = round(entry + sl_d, 2)
            if sl_d > MAX_SL_DISTANCE:
                return None

            # منع فتح بيع إذا كان هناك أوردر بلوك شرائي قريب يعيق الوصول إلى 2.0R على الأقل
            if self.nearest_bull_ob and (entry - self.nearest_bull_ob["top"]) < sl_d * 2.0:
                return None

            candidates = []
            if self.poc_price and self.poc_price < entry:
                candidates.append((self.poc_price + 0.20, "VP_POC"))
            if self.nearest_bull_ob and self.nearest_bull_ob["top"] < entry:
                candidates.append((self.nearest_bull_ob["top"] + 0.20, "BULL_OB_TARGET"))
            if self.top_bid_wall and self.top_bid_wall["price"] < entry:
                candidates.append((self.top_bid_wall["price"] + 0.15, "BID_LIQUIDITY_WALL"))

            valid = [
                (tp, lbl, round((entry - tp) / sl_d, 2))
                for tp, lbl in candidates
                if (entry - tp) >= sl_d * MIN_STRUCTURAL_RR and (entry - tp) <= sl_d * MAX_STRUCTURAL_RR
            ]
            if valid:
                valid.sort(key=lambda x: x[0])
                tp_p, tp_lbl, rr = valid[0]
            else:
                rr = FALLBACK_RR_RATIO
                tp_p = round(entry - sl_d * rr, 2)
                tp_lbl = f"INSTITUTIONAL_TARGET ({rr}R)"

            return raw_sl, tp_p, sl_d, tp_lbl, rr, ob_tag

    async def _open_position(self, decision: Dict):
        if self.balance <= 0 or self.last_price is None:
            return
        side = "BUY" if decision["direction"] == 1 else "SELL"
        half = self.spread / 2.0
        entry = round(self.last_price + half if side == "BUY" else self.last_price - half, 2)

        stops = self._compute_pure_market_structure_stops(side, entry, decision)
        if stops is None:
            self.block_reason = "تم إلغاء الدخول: المسافة الهيكلية للوقف أو الهدف لا تحقق معايير النخبة (1:2.5+)"
            return
        sl_price, tp_price, sl_d, tp_anchor, target_rr, ob_tag = stops
        conf = decision["score"]

        if self.lot_mode == "FIXED":
            lots = round(float(np.clip(self.fixed_lot, 0.01, MAX_LOT_CAP)), 2)
        else:
            risk_usd = self.balance * self.risk_pct
            lots = round(min(MAX_LOT_CAP, max(0.01, risk_usd / max(1.0, sl_d * 100.0))), 2)

        ob_txt = f" | {ob_tag}" if ob_tag else ""
        reason = (
            f"نخبة A+ ({conf}%){ob_txt} | الستوب ({sl_d}$) | "
            f"الهدف: {tp_anchor} ({target_rr}R) | OBI: {self.order_book_imbalance:+.2f}"
        )
        self.open_position = {
            "id": int(time.time() * 1000) % 1000000,
            "side": side,
            "entry_price": entry,
            "current_price": round(self.last_price, 2),
            "sl": sl_price,
            "tp": tp_price,
            "tp_anchor": tp_anchor,
            "ob_tag": ob_tag,
            "target_rr": target_rr,
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
            "obi_val": f"{self.order_book_imbalance:+.2f}",
            "context_key": decision.get("context_key", ""),
            "regime": self.active_regime,
            "reason_entry": reason,
            "opened_at": stamp(),
            "opened_ts": time.time(),
        }
        self._save_runtime_key("open_position", self.open_position)
        self.log_thought(
            f"اقتناص صفقة نخبة A+ [{side}] عند {entry:.2f}$ ({lots:.2f}L) | SL={sl_price}$ | TP={tp_price}$ ({target_rr}R)",
            "EXECUTION",
        )
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
        self._update_drawdown(self.equity)

        r_multiple = diff / max(0.1, pos["initial_risk"])
        pos["r_multiple"] = round(r_multiple, 2)
        held = time.time() - pos["opened_ts"]

        total_tp_dist = max(0.5, abs(pos["tp"] - pos["entry_price"]))
        tp_progress = diff / total_tp_dist

        can_partial = pos["lots"] >= 0.02
        if not pos["be_secured"] and (tp_progress >= 0.65 or r_multiple >= 1.50):
            lock = round(max(self.spread * 1.6 + 0.20, pos["initial_risk"] * 0.20), 2)
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
                    note_msg = f"جني جزئي ({closed_lots:.2f}L) بربح {chunk:+.2f}$ وتأمين الباقي عند {pos['sl']:.2f}$"

            self.log_thought(f"تأمين صفقة النخبة: {note_msg}", "RISK_SHIELD")
            self._save_runtime_key("open_position", pos)
            await self.tg.notify_secured(pos, note_msg)

        hit_tp = q >= pos["tp"] if side == "BUY" else q <= pos["tp"]
        hit_sl = q <= pos["sl"] if side == "BUY" else q >= pos["sl"]
        if hit_tp:
            await self._close_position(q, f"ضرب الهدف المؤسساتي ({pos.get('tp_anchor', 'TP')})")
            return
        if hit_sl:
            reason = "وقف الأرباح المتحرك (BE+)" if pos["be_secured"] else "ضرب الوقف خلف الـ Order Block (SL)"
            await self._close_position(q, reason)
            return

        opposite = (side == "BUY" and decision["direction"] == -1) or (side == "SELL" and decision["direction"] == 1)
        if opposite and decision["score"] >= 86.0 and decision.get("confirmed") and held >= OPPOSITE_EXIT_MIN_HOLD_S:
            await self._close_position(q, f"خروج استباقي: تشكل إعداد نخبة معاكس بقوة {decision['score']}%")
            return

        if held > STAGNATION_EXIT_S and abs(r_multiple) < 0.20 and self.entropy > 0.86:
            await self._close_position(q, f"إلغاء الفرضية ذاتياً: ركود السعر وارتفاع الإنتروبيا لـ {self.entropy}")

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
        self._update_drawdown(self.balance)

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
            delta = float(np.clip(r_result, -1.2, 1.8)) * 0.08
            self.policy_weights[pol] = round(float(np.clip(self.policy_weights[pol] + delta, 0.5, 1.8)), 3)

        self._update_contextual_learning_on_close(pos, total_pnl, r_result, reason_exit)

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
                f"{MAX_CONSECUTIVE_LOSSES} خسائر متتالية: تفعيل درع الحماية وإيقاف الدخول مؤقتاً.",
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
        spawn(self.trigger_background_learning(force=True))

    def _persist_trade_and_lesson(self, rec: Dict, policy_name: str):
        conn = get_db_conn()
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


# ==============================================================================
# 4. محرك البث الحي والفيوتشر لـ MEXC (MEXC Futures & Level 2 Depth Engine)
# ==============================================================================
class MEXCLiveGoldStream:
    def __init__(self, brain_ref: AutonomousQuantitativeBrain):
        self.brain = brain_ref
        self.clients: List[WebSocket] = []
        self.feed_source: str = "CONNECTING_TO_MEXC_FUTURES"
        self.market_open: bool = True
        self.last_poll: float = 0.0
        self.last_good_ts: float = 0.0
        self.hist_loaded: bool = False
        self.last_hist_try: float = 0.0
        self._last_bucket: int = 0
        self._client: Optional[httpx.AsyncClient] = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=6.0, headers={"User-Agent": "Mozilla/5.0"})
        return self._client

    async def _fetch_mexc_ticker(self) -> Optional[Tuple[float, float, float]]:
        try:
            r = await self._http().get(MEXC_TICKER_URL)
            if r.status_code == 200:
                res = r.json()
                if res.get("success") and res.get("data"):
                    d = res["data"]
                    bid = float(d.get("bid1", d.get("lastPrice", 0)))
                    ask = float(d.get("ask1", d.get("lastPrice", 0)))
                    if bid > 100 and ask > 100 and ask >= bid:
                        mid = (bid + ask) / 2.0
                        return bid, ask, mid
        except Exception:
            pass
        return None

    async def _fetch_mexc_depth(self) -> bool:
        try:
            r = await self._http().get(f"{MEXC_DEPTH_URL}?limit=50")
            if r.status_code == 200:
                res = r.json()
                if res.get("success") and res.get("data"):
                    d = res["data"]
                    raw_bids = d.get("bids", [])
                    raw_asks = d.get("asks", [])
                    if raw_bids and raw_asks:
                        bids = [(float(b[0]), float(b[1])) for b in raw_bids[:30]]
                        asks = [(float(a[0]), float(a[1])) for a in raw_asks[:30]]

                        total_bid_vol = sum(b[1] for b in bids)
                        total_ask_vol = sum(a[1] for a in asks)
                        denom = total_bid_vol + total_ask_vol + 1e-6
                        obi = (total_bid_vol - total_ask_vol) / denom

                        top_bid = max(bids, key=lambda x: x[1])
                        top_ask = max(asks, key=lambda x: x[1])

                        self.brain.order_book_imbalance = round(float(obi), 3)
                        self.brain.order_book_bids_vol = round(total_bid_vol, 1)
                        self.brain.order_book_asks_vol = round(total_ask_vol, 1)
                        self.brain.top_bid_wall = {"price": top_bid[0], "volume": round(top_bid[1], 1)}
                        self.brain.top_ask_wall = {"price": top_ask[0], "volume": round(top_ask[1], 1)}
                        return True
        except Exception:
            pass
        return False

    async def _fetch_mexc_history(self) -> List[Dict]:
        try:
            r = await self._http().get(f"{MEXC_KLINE_URL}?interval={MEXC_INTERVAL}")
            if r.status_code == 200:
                res = r.json()
                if res.get("success") and res.get("data"):
                    d = res["data"]
                    times = d.get("time", [])
                    opens = d.get("open", [])
                    highs = d.get("high", [])
                    lows = d.get("low", [])
                    closes = d.get("close", [])
                    vols = d.get("vol", [])

                    n = len(times)
                    if n >= 30:
                        out = []
                        for i in range(n):
                            t = int(times[i])
                            if t > 1e11:
                                t //= 1000
                            out.append({
                                "time": (t // TF_SEC) * TF_SEC,
                                "open": float(opens[i]),
                                "high": float(highs[i]),
                                "low": float(lows[i]),
                                "close": float(closes[i]),
                                "volume": float(vols[i]) if i < len(vols) else 1.0,
                            })
                        return out
        except Exception as e:
            print("MEXC history error:", repr(e))
        return []

    async def _load_history(self):
        self.last_hist_try = time.time()
        hist = await self._fetch_mexc_history()
        if not hist:
            return

        async with self.brain.lock:
            seen = set()
            clean = []
            for c in hist:
                if c["time"] not in seen:
                    seen.add(c["time"])
                    clean.append(c)

            if len(clean) >= 30:
                self.brain.candles.clear()
                self.brain.candles.extend(clean)
                self.brain.save_bulk_candles_to_db(clean)

                closes_arr = np.array([c["close"] for c in self.brain.candles], dtype=float)
                self.brain._warm_history(closes_arr)
                if len(self.brain.candles) > MIN_CANDLES:
                    self.brain._analyze(list(self.brain.candles)[:-1])
                self.brain._analysis_ts = None
                self.hist_loaded = True
                self.brain.log_thought(
                    f"تم تحميل {len(clean)} شمعة فيوتشر حقيقية من MEXC (مع الفوليوم الفعلي والـ OBs العذراء).",
                    "PERCEPTION",
                )
                spawn(self.brain.trigger_background_learning(force=True))

        if self.clients:
            await self._broadcast(self.build_payload("INIT_SNAPSHOT", full=True))

    def _update_candle(self, price: float, bucket: int):
        candles = self.brain.candles
        if not candles or bucket > candles[-1]["time"]:
            if candles:
                self.brain.save_candle_to_db(candles[-1])
            candles.append({
                "time": bucket, "open": price, "high": price,
                "low": price, "close": price, "volume": 1.0
            })
            return
        last = candles[-1]
        last["close"] = price
        if price > last["high"]:
            last["high"] = price
        if price < last["low"]:
            last["low"] = price
        if not last.get("vol_real"):
            last["volume"] = last.get("volume", 0.0) + 1.0

    async def _sync_recent_volume(self):
        fresh = await self._fetch_mexc_history()
        if not fresh:
            return
        by_time = {c["time"]: c for c in fresh[-12:]}
        async with self.brain.lock:
            candles = self.brain.candles
            for i, c in enumerate(candles):
                src = by_time.get(c["time"])
                if not src:
                    continue
                c["volume"] = src["volume"]
                c["vol_real"] = True
                if i < len(candles) - 1:
                    c["open"], c["high"], c["low"], c["close"] = src["open"], src["high"], src["low"], src["close"]
                    self.brain.save_candle_to_db(c)

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
        if bucket != self._last_bucket:
            if self._last_bucket and self.hist_loaded:
                spawn(self._sync_recent_volume())
            self._last_bucket = bucket

        if now - self.last_poll >= POLL_S:
            self.last_poll = now
            ticker = await self._fetch_mexc_ticker()
            if ticker:
                bid, ask, mid = ticker
                self.last_good_ts = now
                self.market_open = True
                self.feed_source = "MEXC FUTURES (XAU_USDT)"
                b.bid_price = round(bid, 2)
                b.ask_price = round(ask, 2)
                b.last_price = round(mid, 2)
                b.spread = round(max(ask - bid, 0.0), 2)
                self._update_candle(mid, bucket)

                await self._fetch_mexc_depth()

                if not self.hist_loaded and now - self.last_hist_try > 10.0:
                    await self._load_history()

        fresh = (now - self.last_good_ts) < 20.0
        b.can_trade = fresh and self.market_open

        await b.process_market_state()
        await self.broadcast_state()

    def build_payload(self, ptype: str, full: bool = False) -> Dict:
        b = self.brain
        now_sec = int(time.time())
        candle_rem_s = TF_SEC - (now_sec % TF_SEC)
        payload = {
            "type": ptype,
            "tf_min": TF_MIN,
            "candle_rem_s": candle_rem_s,
            "feed_source": self.feed_source,
            "session": get_active_session(),
            "liquidity_ok": b.liquidity_ok,
            "volume_ratio": b.volume_ratio,
            "bot_active": b.bot_active,
            "lot_mode": b.lot_mode,
            "fixed_lot": b.fixed_lot,
            "risk_pct": round(b.risk_pct * 100.0, 2),
            "price": b.last_price,
            "bid": b.bid_price,
            "ask": b.ask_price,
            "spread": b.spread,
            "max_spread": MAX_SPREAD,
            "balance": b.balance,
            "equity": b.equity,
            "initial_balance": b.initial_balance,
            "kpis": b.get_kpi_metrics(),
            "depth": {
                "obi": b.order_book_imbalance,
                "bid_vol": b.order_book_bids_vol,
                "ask_vol": b.order_book_asks_vol,
                "top_bid_wall": b.top_bid_wall,
                "top_ask_wall": b.top_ask_wall,
            },
            "order_blocks": b.order_blocks,
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
                "ema_fast": b.ema_fast,
                "ema_slow": b.ema_slow,
                "macro_bias": b.macro_bias,
                "swing_high": b.nearest_swing_high,
                "swing_low": b.nearest_swing_low,
                "range_pos": b.range_position_pct,
                "poc": b.poc_price,
                "vah": b.vah_price,
                "val": b.val_price,
                "smc_event": b.smc_event,
                "regime": b.active_regime,
                "policy": b.active_policy,
                "confidence": b.confidence,
                "entry_threshold": b.adaptive_threshold,
                "sl_buffer_mult": b.sl_buffer_mult,
                "ml_status": b.auto_ml.status_text,
                "ml_probs": b.ml_probs,
                "weights": b.policy_weights,
                "block_reason": b.block_reason,
            },
            "discovered_zones": b.discovered_zones,
            "open_position": b.open_position,
            "trade_log": b.trade_log[:30],
            "thoughts": list(b.thoughts)[:35],
        }
        if full:
            payload["candles"] = list(b.candles)[-350:]
            payload["ema_fast_series"] = b.ema_fast_series[-350:]
            payload["ema_slow_series"] = b.ema_slow_series[-350:]
        else:
            payload["candle"] = b.candles[-1] if b.candles else None
            payload["ema_fast_point"] = b.ema_fast_series[-1] if b.ema_fast_series else None
            payload["ema_slow_point"] = b.ema_slow_series[-1] if b.ema_slow_series else None
        return clean_json(payload)

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


stream_manager = MEXCLiveGoldStream(brain)


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


app = FastAPI(title="Autonomous XAU/USD Institutional A+ Sniper Cockpit v8", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if DASHBOARD_TOKEN and not hmac.compare_digest(
        str(ws.query_params.get("token", "")).encode(), DASHBOARD_TOKEN.encode()
    ):
        await ws.close(code=1008)
        return
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
                try:
                    if cmd == "TOGGLE_BOT":
                        brain.bot_active = not brain.bot_active
                        state_txt = "مُفعّل ومستقل" if brain.bot_active else "مُتوقف مؤقتاً"
                        brain.log_thought(f"تم تغيير حالة الوكيل يدوياً إلى: {state_txt}", "SYSTEM")

                    elif cmd == "TOGGLE_LOT_MODE":
                        new_mode = "DYNAMIC_RISK" if brain.lot_mode == "FIXED" else "FIXED"
                        brain.update_lot_settings(lot_mode=new_mode)
                        mode_lbl = f"{brain.fixed_lot:.2f} لوت ثابت" if brain.lot_mode == "FIXED" else f"مخاطرة ديناميكية ({brain.risk_pct*100:.2f}%)"
                        brain.log_thought(f"تم تبديل وضع العقد إلى: {mode_lbl}", "SYSTEM")

                    elif cmd == "ADJUST_LOT":
                        step = float(data.get("step", 0.0))
                        exact = data.get("exact")
                        new_lot = float(exact) if exact is not None else round(brain.fixed_lot + step, 2)
                        brain.update_lot_settings(lot_mode="FIXED", fixed_lot=new_lot)
                        brain.log_thought(f"تم ضبط حجم اللوت الثابت على: {brain.fixed_lot:.2f} Lot", "SYSTEM")

                    elif cmd == "SET_RISK_PCT":
                        pct = float(data.get("risk_pct", 0.5)) / 100.0
                        brain.update_lot_settings(lot_mode="DYNAMIC_RISK", risk_pct=pct)
                        brain.log_thought(f"تم تفعيل المخاطرة الديناميكية بنسبة: {brain.risk_pct*100:.2f}%", "SYSTEM")

                    elif cmd == "PANIC_CLOSE" and brain.open_position is not None:
                        await brain._close_position(
                            brain._exit_quote(brain.open_position["side"]),
                            "إغلاق فوري بطلب المالك (Manual Emergency Exit)",
                        )

                    elif cmd == "RELOAD_HISTORY":
                        stream_manager.hist_loaded = False
                        spawn(stream_manager._load_history())
                        brain.log_thought("جاري إعادة مزامنة شموع فيوتشر MEXC والأوردر بلوك...", "SYSTEM")

                    elif cmd == "TRAIN_NOW":
                        spawn(brain.trigger_background_learning(force=True))
                        brain.log_thought("بدأ التدريب الفوري لنموذج Auto-ML على بيانات فيوتشر MEXC...", "SYSTEM")

                    elif cmd == "SAVE_TELEGRAM":
                        token = str(data.get("bot_token", "")).strip()
                        chat_id = str(data.get("chat_id", "")).strip()
                        enabled = bool(data.get("enabled", True))
                        brain.update_telegram_config(token, chat_id, enabled)
                        brain.log_thought(f"تم تحديث إعدادات تلغرام ({brain.tg.last_status}).", "SYSTEM")

                    elif cmd == "TEST_TELEGRAM":
                        test_msg = (
                            "🔔 <b>اختبار اتصال ناجح | XAU/USD Institutional A+ Sniper v8</b>\n"
                            "━━━━━━━━━━━━━━━━━━\n"
                            f"✅ البوت متصل بقناتك بنجاح (مصفاة السيولة الصارمة وصفقات النخبة A+).\n"
                            f"📡 <b>المصدر:</b> <code>MEXC Futures (XAU_USDT)</code>\n"
                            f"💰 <b>الرصيد:</b> <code>${brain.balance:,.2f}</code>\n"
                            f"📊 <b>سعر الذهب:</b> <code>${brain.last_price or 0.0:.2f}</code>\n"
                            f"⚖️ <b>ميزان السيولة (OBI):</b> <code>{brain.order_book_imbalance:+.2f}</code>\n"
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
                            "تم إرسال التقرير اليومي إلى تلغرام." if ok else f"تعذر الإرسال: {brain.tg.last_status}",
                            "SYSTEM",
                        )
                except (ValueError, TypeError) as e:
                    brain.log_thought(f"أمر مرفوض ({cmd}): قيمة غير صالحة — {e}", "SYSTEM")

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
  <title>XAU/USD — Institutional A+ Order Block Cockpit v8</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/react@18.3.1/umd/react.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/@babel/standalone@7.26.4/babel.min.js"></script>
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-obsidian: #04060A;
      --card-surface: rgba(11, 15, 24, 0.78);
      --border-specular: rgba(255, 255, 255, 0.09);
    }
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    body {
      background-color: var(--bg-obsidian);
      color: #F8FAFC;
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
    .vision-card {
      background: linear-gradient(165deg, rgba(20, 26, 40, 0.82) 0%, rgba(9, 12, 20, 0.90) 100%);
      backdrop-filter: blur(24px);
      -webkit-backdrop-filter: blur(24px);
      border: 1px solid var(--border-specular);
      border-top: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 20px;
      box-shadow: 0 20px 50px -12px rgba(0, 0, 0, 0.75), inset 0 1px 0 0 rgba(255, 255, 255, 0.08);
    }
    .vision-pill {
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.018) 100%);
      border: 1px solid rgba(255, 255, 255, 0.085);
      border-top: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 14px;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .apple-btn {
      transition: transform 0.12s ease, background-color 0.2s ease, border-color 0.2s ease;
    }
    .apple-btn:active {
      transform: scale(0.95);
    }
    .gold-gradient-text {
      background: linear-gradient(135deg, #FDE68A 0%, #F59E0B 55%, #D97706 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .no-scrollbar::-webkit-scrollbar { display: none; }
    .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
    @keyframes pulseRing {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.45; transform: scale(0.90); }
    }
    .dot-live { animation: pulseRing 2s infinite ease-in-out; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.22); border-radius: 9999px; }
  </style>
</head>
<body>
  <div id="root"><div id="boot" style="padding:24px;color:#94A3B8;font-family:sans-serif;direction:rtl">جاري تحميل الواجهة...</div></div>
  <script>
    window.addEventListener('error', function (ev) {
      var r = document.getElementById('boot');
      if (r) r.textContent = 'خطأ في الواجهة: ' + (ev.message || 'غير معروف');
    });
    setTimeout(function () {
      var r = document.getElementById('boot');
      if (!r) return;
      var miss = [];
      if (!window.React) miss.push('React');
      if (!window.ReactDOM) miss.push('ReactDOM');
      if (!window.Babel) miss.push('Babel');
      if (!window.LightweightCharts) miss.push('LightweightCharts');
      if (!window.tailwind) miss.push('Tailwind');
      r.textContent = miss.length
        ? 'تعذّر تحميل المكتبات من الإنترنت: ' + miss.join(', ') + ' — تحقق من الشبكة/حاجب الإعلانات.'
        : 'المكتبات تحمّلت لكن الواجهة لم تُرسم — افتح Console (F12) لرؤية الخطأ.';
    }, 8000);
  </script>

  <script type="text/babel">
    const { useState, useEffect, useRef, useMemo } = React;

    const num = (v) => (v === null || v === undefined || v === '' ? NaN : Number(v));

    const cleanCandles = (arr) => {
      const rows = (Array.isArray(arr) ? arr : []).map(c => c && ({
        time: num(c.time), open: num(c.open), high: num(c.high),
        low: num(c.low), close: num(c.close),
      })).filter(c => c && [c.time, c.open, c.high, c.low, c.close].every(Number.isFinite));
      rows.sort((a, b) => a.time - b.time);
      const out = [];
      rows.forEach(c => { if (!out.length || c.time > out[out.length - 1].time) out.push(c); });
      return out;
    };

    const cleanLine = (arr) => {
      const rows = (Array.isArray(arr) ? arr : []).map(p => p && ({ time: num(p.time), value: num(p.value) }))
        .filter(p => p && Number.isFinite(p.time) && Number.isFinite(p.value));
      rows.sort((a, b) => a.time - b.time);
      const out = [];
      rows.forEach(p => { if (!out.length || p.time > out[out.length - 1].time) out.push(p); });
      return out;
    };

    class ErrorBoundary extends React.Component {
      constructor(props) { super(props); this.state = { err: null }; }
      static getDerivedStateFromError(err) { return { err }; }
      componentDidCatch(err, info) { console.error('UI error:', err, info); }
      render() {
        if (this.state.err) {
          return (
            <div style={{ padding: 16, margin: 8, border: '1px solid #F43F5E', borderRadius: 12, color: '#FDA4AF', direction: 'ltr', fontFamily: 'monospace', fontSize: 12 }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>UI error in: {this.props.label || 'component'}</div>
              <pre style={{ whiteSpace: 'pre-wrap' }}>{String(this.state.err && (this.state.err.stack || this.state.err.message || this.state.err))}</pre>
              <button onClick={() => this.setState({ err: null })} style={{ marginTop: 8, padding: '4px 10px', border: '1px solid #F43F5E', borderRadius: 8 }}>إعادة المحاولة</button>
            </div>
          );
        }
        return this.props.children;
      }
    }

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

    function EquitySparkline({ initialBalance, currentEquity, tradeLog }) {
      const points = useMemo(() => {
        const logs = [...(tradeLog || [])].reverse();
        let cur = Number(initialBalance || 1000);
        const vals = [cur];
        logs.forEach(t => {
          cur += Number(t.pnl || 0);
          vals.push(cur);
        });
        if (currentEquity !== undefined && currentEquity !== null) {
          vals.push(Number(currentEquity));
        }
        return vals;
      }, [initialBalance, currentEquity, tradeLog]);

      if (points.length < 2) return null;
      const minV = Math.min(...points);
      const maxV = Math.max(...points);
      const span = Math.max(0.5, maxV - minV);
      const w = 96, h = 28;
      const coords = points.map((v, idx) => {
        const x = (idx / (points.length - 1)) * w;
        const y = h - 4 - ((v - minV) / span) * (h - 8);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      const isUp = points[points.length - 1] >= points[0];
      const strokeColor = isUp ? "#10B981" : "#F43F5E";

      return (
        <svg width={w} height={h} className="overflow-visible shrink-0">
          <polyline
            fill="none"
            stroke={strokeColor}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={coords.join(" ")}
          />
        </svg>
      );
    }

    function RadialConvictionGauge({ confidence, threshold = 84, macroBias }) {
      const r = 34;
      const c = 2 * Math.PI * r;
      const pct = Math.min(100, Math.max(0, Number(confidence || 0)));
      const offset = c - (pct / 100) * c;
      const isReady = pct >= threshold;
      const strokeColor = isReady ? "#10B981" : pct >= 70 ? "#F59E0B" : "#38BDF8";

      return (
        <div className="relative flex items-center justify-center w-24 h-24 shrink-0">
          <svg className="w-24 h-24 -rotate-90" viewBox="0 0 84 84">
            <circle cx="42" cy="42" r={r} stroke="rgba(255,255,255,0.07)" strokeWidth="6.5" fill="none" />
            <circle
              cx="42"
              cy="42"
              r={r}
              stroke={strokeColor}
              strokeWidth="6.5"
              strokeDasharray={c}
              strokeDashoffset={offset}
              strokeLinecap="round"
              fill="none"
              style={{ transition: "stroke-dashoffset 0.45s ease, stroke 0.3s ease" }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            <span className="mono text-base font-extrabold text-white">{pct.toFixed(1)}%</span>
            <span className={`text-[9px] font-bold tracking-wider ${
              macroBias === 'BULLISH' ? 'text-emerald-400' : macroBias === 'BEARISH' ? 'text-rose-400' : 'text-amber-300'
            }`}>
              {macroBias}
            </span>
          </div>
        </div>
      );
    }

    function TradingChart({
      candles, latestCandle, zones, openPos, poc, vah, val,
      orderBlocks, swingHigh, swingLow,
      emaFastSeries, emaSlowSeries, emaFastPoint, emaSlowPoint,
      showOB, showVP, showEMA
    }) {
      const containerRef = useRef(null);
      const chartRef = useRef(null);
      const seriesRef = useRef(null);
      const emaFastRef = useRef(null);
      const emaSlowRef = useRef(null);
      const obLinesRef = useRef([]);
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
            vertLines: { color: 'rgba(255, 255, 255, 0.025)' },
            horzLines: { color: 'rgba(255, 255, 255, 0.025)' },
          },
          crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: { color: 'rgba(245, 158, 11, 0.40)', width: 1, style: 2 },
            horzLine: { color: 'rgba(245, 158, 11, 0.40)', width: 1, style: 2 },
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

        const emaFastLine = chart.addLineSeries({
          color: '#38BDF8',
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
        });

        const emaSlowLine = chart.addLineSeries({
          color: '#F59E0B',
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Dashed,
          priceLineVisible: false,
          lastValueVisible: false,
        });

        chartRef.current = chart;
        seriesRef.current = series;
        emaFastRef.current = emaFastLine;
        emaSlowRef.current = emaSlowLine;

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
          seriesRef.current.setData(cleanCandles(candles));
          chartRef.current.timeScale().fitContent();
        }
      }, [candles]);

      useEffect(() => {
        if (seriesRef.current && latestCandle) {
          try { const lc = cleanCandles([latestCandle]); if (lc.length) seriesRef.current.update(lc[0]); } catch (e) { console.warn("candle update skipped", e); }
        }
      }, [latestCandle]);

      useEffect(() => {
        if (!emaFastRef.current || !emaSlowRef.current) return;
        if (!showEMA) {
          emaFastRef.current.setData([]);
          emaSlowRef.current.setData([]);
          return;
        }
        if (emaFastSeries && emaFastSeries.length > 0) emaFastRef.current.setData(cleanLine(emaFastSeries));
        if (emaSlowSeries && emaSlowSeries.length > 0) emaSlowRef.current.setData(cleanLine(emaSlowSeries));
      }, [emaFastSeries, emaSlowSeries, showEMA]);

      useEffect(() => {
        if (!showEMA || !emaFastRef.current || !emaSlowRef.current) return;
        try {
          const fp = cleanLine([emaFastPoint]); if (fp.length) emaFastRef.current.update(fp[0]);
          const sp = cleanLine([emaSlowPoint]); if (sp.length) emaSlowRef.current.update(sp[0]);
        } catch (e) { console.warn("ema update skipped", e); }
      }, [emaFastPoint, emaSlowPoint, showEMA]);

      useEffect(() => {
        if (!seriesRef.current) return;
        obLinesRef.current.forEach(l => seriesRef.current.removePriceLine(l));
        obLinesRef.current = [];
        if (!showOB || !orderBlocks) return;

        orderBlocks.filter(ob => ob && Number.isFinite(ob.top) && Number.isFinite(ob.bottom)).forEach((ob, idx) => {
          const isBull = ob.type === "BULLISH_OB";
          const color = isBull ? "rgba(16, 185, 129, 0.85)" : "rgba(244, 63, 94, 0.85)";
          const lineTop = seriesRef.current.createPriceLine({
            price: ob.top,
            color: color,
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Solid,
            axisLabelVisible: !openPos && idx < 3,
            title: isBull ? `BULL OB (${ob.status === 'TESTED' ? 'ملموس' : 'عذراء A+'})` : '',
          });
          const lineBot = seriesRef.current.createPriceLine({
            price: ob.bottom,
            color: color,
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: !openPos && idx < 3,
            title: !isBull ? `BEAR OB (${ob.status === 'TESTED' ? 'ملموس' : 'عذراء A+'})` : '',
          });
          obLinesRef.current.push(lineTop, lineBot);
        });
      }, [orderBlocks, showOB, openPos]);

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
            title: 'POC (MEXC)',
          }));
        }
      }, [poc, showVP]);

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
          title: `TP (${openPos.target_rr || 2.5}R)`,
        }));
        posLinesRef.current.push(seriesRef.current.createPriceLine({
          price: openPos.sl,
          color: '#F43F5E',
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Dotted,
          axisLabelVisible: true,
          title: openPos.be_secured ? 'SL (BE+)' : 'OB SL',
        }));
      }, [openPos]);

      return <div ref={containerRef} className="w-full h-[280px] sm:h-[370px] lg:h-[445px]" />;
    }

    function App() {
      const [connected, setConnected] = useState(false);
      const [snapshotCandles, setSnapshotCandles] = useState([]);
      const [latestCandle, setLatestCandle] = useState(null);
      const [emaFastSeries, setEmaFastSeries] = useState([]);
      const [emaSlowSeries, setEmaSlowSeries] = useState([]);
      const [emaFastPoint, setEmaFastPoint] = useState(null);
      const [emaSlowPoint, setEmaSlowPoint] = useState(null);

      const [state, setState] = useState({
        tf_min: 1,
        candle_rem_s: 60,
        feed_source: "CONNECTING TO MEXC...",
        session: { name: "INIT", label: "جاري تحديد الجلسة...", bias: "NEUTRAL", high_liquidity: false },
        liquidity_ok: false,
        volume_ratio: 1.0,
        bot_active: true,
        lot_mode: "FIXED",
        fixed_lot: 0.01,
        risk_pct: 0.50,
        price: null,
        bid: null,
        ask: null,
        spread: 0.25,
        max_spread: 0.99,
        balance: 1000,
        equity: 1000,
        initial_balance: 1000,
        kpis: { win_rate: 0, total_trades: 0, wins: 0, losses: 0, profit_factor: 0, net_pnl: 0, expectancy: 0, max_drawdown_pct: 0 },
        depth: { obi: 0.0, bid_vol: 0, ask_vol: 0, top_bid_wall: null, top_ask_wall: null },
        order_blocks: [],
        telegram: { enabled: false, configured: false, masked_token: "", chat_id: "", last_status: "غير مرتبط", last_report_date: "" },
        cognition: {
          hurst: 0.5, hurst_rank: 50, entropy: 0.5, entropy_rank: 50,
          velocity: 0, sigma: 0, atr: 1.5, z_score: 0,
          ema_fast: null, ema_slow: null, macro_bias: "NEUTRAL",
          swing_high: null, swing_low: null, range_pos: 50,
          poc: null, vah: null, val: null, smc_event: "NONE",
          regime: "INITIALIZING", policy: "OBSERVING", confidence: 0,
          entry_threshold: 84.0,
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

      const [showOB, setShowOB] = useState(true);
      const [showVP, setShowVP] = useState(true);
      const [showEMA, setShowEMA] = useState(true);

      const [tgTokenInput, setTgTokenInput] = useState("");
      const [tgChatInput, setTgChatInput] = useState("");
      const [tgEnabledInput, setTgEnabledInput] = useState(true);
      const tgLoadedRef = useRef(false);

      useEffect(() => {
        let timer;
        function connect() {
          const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
          const tk = new URLSearchParams(window.location.search).get('token') || '';
          const ws = new WebSocket(`${proto}//${window.location.host}/ws${tk ? '?token=' + encodeURIComponent(tk) : ''}`);
          wsRef.current = ws;

          ws.onopen = () => setConnected(true);
          ws.onmessage = (e) => {
            let data;
            try { data = JSON.parse(e.data); } catch (err) { console.warn("bad WS payload", err); return; }
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
              if (data.ema_fast_series) setEmaFastSeries(data.ema_fast_series);
              if (data.ema_slow_series) setEmaSlowSeries(data.ema_slow_series);
              setState(prev => ({ ...prev, ...data }));
            } else if (data.type === "TICK_UPDATE") {
              if (data.candle) setLatestCandle(data.candle);
              if (data.ema_fast_point) setEmaFastPoint(data.ema_fast_point);
              if (data.ema_slow_point) setEmaSlowPoint(data.ema_slow_point);
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

      const { cognition: cog, kpis, open_position: pos, session, telegram: tg, depth } = state;

      const formatCountdown = (sec) => {
        const m = Math.floor(sec / 60);
        const s = sec % 60;
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
      };

      const renderActivePosition = () => {
        if (!pos) {
          return (
            <div className="py-5 text-center text-xs text-slate-400">
              لا توجد صفقة نشطة — البوت يراقب بصرامة أوقات السيولة العالية لاقتناص صفقات النخبة (A+ Unmitigated Order Blocks) فقط.
            </div>
          );
        }
        return (
          <div className="space-y-3">
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-xs">
              <div className="vision-pill p-2.5">
                <span className="text-[10px] text-slate-400 block">الاتجاه / اللوت</span>
                <span className={`font-extrabold mono text-sm ${pos.side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {pos.side} ({pos.lots}L)
                </span>
              </div>
              <div className="vision-pill p-2.5">
                <span className="text-[10px] text-slate-400 block">سعر الدخول</span>
                <span className="font-bold mono text-sm text-sky-300">${pos.entry_price.toFixed(2)}</span>
              </div>
              <div className="vision-pill p-2.5">
                <span className="text-[10px] text-slate-400 block">السعر الحالي</span>
                <span className="font-bold mono text-sm text-white">${pos.current_price.toFixed(2)}</span>
              </div>
              <div className="vision-pill p-2.5">
                <span className="text-[10px] text-slate-400 block">الوقف خلف الـ OB</span>
                <span className="font-bold mono text-sm text-rose-400">${pos.sl.toFixed(2)}</span>
              </div>
              <div className="vision-pill p-2.5">
                <span className="text-[10px] text-slate-400 block">الهدف ({pos.target_rr || 2.5}R)</span>
                <span className="font-bold mono text-sm text-emerald-400">${pos.tp.toFixed(2)}</span>
              </div>
              <div className="vision-pill p-2.5">
                <span className="text-[10px] text-slate-400 block">العائد اللحظي (R)</span>
                <span className={`font-bold mono text-sm ${(pos.r_multiple || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                  {(pos.r_multiple || 0) >= 0 ? '+' : ''}{Number(pos.r_multiple || 0).toFixed(2)}R
                </span>
              </div>
              <div className="vision-pill p-2.5 col-span-2 sm:col-span-1">
                <span className="text-[10px] text-slate-400 block">الربح اللحظي</span>
                <span className={`font-extrabold mono text-sm ${pos.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                </span>
              </div>
            </div>

            <div className="vision-pill p-3">
              <div className="flex items-center justify-between text-[11px] mb-1.5">
                <span className="text-rose-400 mono font-semibold">SL: ${pos.sl.toFixed(2)}</span>
                <span className="text-slate-300 text-[11px]">
                  {pos.be_secured ? '🛡️ مؤمّنة (Break-Even+)' : `🎯 المحطة: ${pos.tp_anchor || 'INSTITUTIONAL_TP'}`}
                </span>
                <span className="text-emerald-400 mono font-semibold">TP: ${pos.tp.toFixed(2)}</span>
              </div>
            </div>
          </div>
        );
      };

      const renderCortexPanel = () => (
        <div className="vision-card p-4 flex flex-col gap-3.5">
          <div className="flex items-center justify-between gap-3">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${state.liquidity_ok ? 'bg-emerald-400' : 'bg-rose-500'}`}></span>
                <h2 className="text-xs font-extrabold gold-gradient-text tracking-wide">مصفاة السيولة ورادار النخبة A+</h2>
              </div>
              <p className="text-[11px] text-slate-300 mt-1.5 leading-relaxed">{cog.block_reason}</p>
            </div>
            <RadialConvictionGauge confidence={cog.confidence} threshold={cog.entry_threshold || 84} macroBias={cog.macro_bias} />
          </div>

          <div className="vision-pill p-3 space-y-1.5">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-emerald-400 font-bold">شراء: {depth.bid_vol} عقد</span>
              <span className="mono font-extrabold text-amber-300">OBI: {depth.obi > 0 ? '+' : ''}{depth.obi}</span>
              <span className="text-rose-400 font-bold">بيع: {depth.ask_vol} عقد</span>
            </div>
            <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden flex">
              <div
                className="bg-emerald-500 transition-all duration-300"
                style={{ width: `${Math.max(5, Math.min(95, ((depth.obi + 1) / 2) * 100))}%` }}
              />
              <div className="bg-rose-500 flex-1" />
            </div>
            <div className="flex items-center justify-between text-[10px] text-slate-400 mono">
              <span>حجم الشمعة: {Number(state.volume_ratio || 1).toFixed(2)}x</span>
              <span>حالة السيولة: {state.liquidity_ok ? '✅ نشطة' : '🚫 محظورة/ضعيفة'}</span>
            </div>
          </div>

          <div>
            <div className="text-[11px] text-slate-400 mb-1.5 font-semibold flex items-center justify-between">
              <span>مناطق الـ Order Blocks النشطة:</span>
              <span className="mono text-amber-300">{state.order_blocks.length} بلوكات</span>
            </div>
            <div className="grid grid-cols-2 gap-1.5 text-xs">
              {state.order_blocks.slice(0, 4).map((ob, idx) => (
                <div key={idx} className="vision-pill px-2.5 py-1.5 flex flex-col justify-between">
                  <div className="flex items-center justify-between">
                    <span className={`mono font-bold text-[11px] ${ob.type === 'BULLISH_OB' ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {ob.type === 'BULLISH_OB' ? 'طلب (Bull OB)' : 'عرض (Bear OB)'}
                    </span>
                    <span className="text-[9px] text-slate-400">{ob.status}</span>
                  </div>
                  <span className="mono text-[10px] text-slate-200 mt-0.5">
                    ${ob.bottom.toFixed(2)} - ${ob.top.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="vision-pill p-2.5">
              <div className="flex items-center justify-between text-[10px] text-slate-400">
                <span>هيرست (Hurst)</span>
                <span className="mono text-emerald-400">{cog.hurst_rank}%</span>
              </div>
              <div className="font-bold text-emerald-300 mt-0.5 mono text-sm">{Number(cog.hurst).toFixed(3)}</div>
            </div>

            <div className="vision-pill p-2.5">
              <div className="flex items-center justify-between text-[10px] text-slate-400">
                <span>إنتروبيا شانون</span>
                <span className="mono text-amber-400">{cog.entropy_rank}%</span>
              </div>
              <div className="font-bold text-amber-300 mt-0.5 mono text-sm">{Number(cog.entropy).toFixed(3)}</div>
            </div>
          </div>

          <div className="vision-pill p-2.5 flex flex-wrap items-center justify-between gap-2 text-[11px]">
            <span className="text-slate-400 font-semibold">Volume Profile (MEXC):</span>
            <div className="flex items-center gap-2.5 mono">
              <span className="text-purple-300">POC: <b>${cog.poc ? cog.poc.toFixed(2) : '--'}</b></span>
              <span className="text-slate-300">VAL: <b>${cog.val ? cog.val.toFixed(2) : '--'}</b></span>
              <span className="text-slate-300">VAH: <b>${cog.vah ? cog.vah.toFixed(2) : '--'}</b></span>
            </div>
          </div>
        </div>
      );

      const renderControlPanel = () => (
        <div className="vision-card p-4 flex flex-col gap-3">
          <h2 className="text-xs font-extrabold gold-gradient-text tracking-wide">لوحة التحكم</h2>

          <div className="vision-pill p-3 space-y-2">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-slate-300">وضع العقد</span>
              <button onClick={() => sendCommand('TOGGLE_LOT_MODE')}
                className="apple-btn px-2.5 py-1 rounded-lg text-[11px] font-bold bg-white/5 border border-white/10 text-amber-300">
                {state.lot_mode === 'FIXED' ? 'لوت ثابت' : 'مخاطرة ديناميكية'}
              </button>
            </div>
            {state.lot_mode === 'FIXED' ? (
              <div className="flex items-center justify-between">
                <button onClick={() => sendCommand('ADJUST_LOT', { step: -0.01 })} className="apple-btn w-9 h-8 rounded-lg bg-white/5 border border-white/10 font-bold">-</button>
                <span className="mono font-extrabold text-amber-300">{Number(state.fixed_lot).toFixed(2)} Lot</span>
                <button onClick={() => sendCommand('ADJUST_LOT', { step: 0.01 })} className="apple-btn w-9 h-8 rounded-lg bg-white/5 border border-white/10 font-bold">+</button>
              </div>
            ) : (
              <div className="flex items-center justify-between">
                <button onClick={() => sendCommand('SET_RISK_PCT', { risk_pct: Math.max(0.2, Number(state.risk_pct) - 0.1) })} className="apple-btn w-9 h-8 rounded-lg bg-white/5 border border-white/10 font-bold">-</button>
                <span className="mono font-extrabold text-amber-300">{Number(state.risk_pct).toFixed(2)}%</span>
                <button onClick={() => sendCommand('SET_RISK_PCT', { risk_pct: Math.min(3, Number(state.risk_pct) + 0.1) })} className="apple-btn w-9 h-8 rounded-lg bg-white/5 border border-white/10 font-bold">+</button>
              </div>
            )}
          </div>

          <button onClick={() => sendCommand('TRAIN_NOW')}
            className="apple-btn px-3 py-2 rounded-xl text-[11px] font-bold bg-white/5 border border-white/10 text-sky-300">
            🧠 تدريب Auto-ML الآن
          </button>

          <form onSubmit={saveTelegramSettings} className="vision-pill p-3 space-y-2">
            <div className="text-[11px] font-bold text-slate-200">إعدادات تلغرام</div>
            <input type="password" autoComplete="off" value={tgTokenInput} onChange={(e) => setTgTokenInput(e.target.value)}
              placeholder={tg.masked_token ? `التوكن الحالي: ${tg.masked_token}` : 'Bot Token'}
              className="w-full px-2.5 py-1.5 rounded-lg bg-black/30 border border-white/10 text-[11px] mono text-slate-100 outline-none" />
            <input type="text" value={tgChatInput} onChange={(e) => setTgChatInput(e.target.value)}
              placeholder="Chat ID"
              className="w-full px-2.5 py-1.5 rounded-lg bg-black/30 border border-white/10 text-[11px] mono text-slate-100 outline-none" />
            <label className="flex items-center gap-2 text-[11px] text-slate-300">
              <input type="checkbox" checked={tgEnabledInput} onChange={(e) => setTgEnabledInput(e.target.checked)} />
              تفعيل الإشعارات
            </label>
            <div className="flex items-center gap-1.5 flex-wrap">
              <button type="submit" className="apple-btn px-2.5 py-1.5 rounded-lg text-[11px] font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">حفظ</button>
              <button type="button" onClick={() => sendCommand('TEST_TELEGRAM')} className="apple-btn px-2.5 py-1.5 rounded-lg text-[11px] font-bold bg-white/5 text-slate-200 border border-white/10">اختبار</button>
              <button type="button" onClick={() => sendCommand('SEND_DAILY_REPORT_NOW')} className="apple-btn px-2.5 py-1.5 rounded-lg text-[11px] font-bold bg-white/5 text-slate-200 border border-white/10">تقرير اليوم</button>
            </div>
            <p className="text-[10px] text-slate-400">{tg.last_status}</p>
          </form>
        </div>
      );

      const renderNeuralStream = () => (
        <div className="vision-card p-4 flex-1 flex flex-col min-h-[260px]">
          <div className="flex items-center justify-between pb-2.5 mb-2.5 border-b border-white/[0.06]">
            <h3 className="text-xs font-extrabold text-slate-200">مجرى التفكير المؤسساتي الحي (Neural Stream)</h3>
            <span className="text-[10px] text-emerald-400 mono">A+ STRICT</span>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 pr-1 text-xs max-h-[340px]">
            {state.thoughts.length === 0 ? (
              <div className="text-slate-500 text-center py-8">بانتظار تدفق السيولة لتحليل السوق...</div>
            ) : (
              state.thoughts.map((th, i) => (
                <div key={i} className="vision-pill p-2.5 transition hover:border-white/20">
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
        <div className="min-h-screen transition-all duration-700 bg-[#04060A]">
          <div className="min-h-screen flex flex-col p-2.5 sm:p-3.5 lg:p-4 pb-24 lg:pb-4 gap-3.5 max-w-[1920px] mx-auto select-none">

            <header className="vision-card p-3 sm:p-3.5 flex flex-col gap-2.5">
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-3">
                  <div className="relative flex items-center justify-center w-9 h-9 rounded-2xl bg-amber-500/10 border border-amber-500/30 shrink-0 shadow-inner">
                    <span className={`w-2.5 h-2.5 rounded-full ${connected ? 'bg-emerald-400 dot-live' : 'bg-rose-500'}`}></span>
                  </div>
                  <div>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-black text-base sm:text-lg tracking-tight gold-gradient-text mono">XAU_USDT</span>
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/30">
                        A+ SNIPER v8
                      </span>
                      <span className="hidden sm:inline-block text-[10px] px-2.5 py-0.5 rounded-full bg-white/5 text-slate-300 border border-white/10">
                        {session?.label}
                      </span>
                    </div>
                    <p className="text-[10px] text-slate-400 mono">{state.feed_source}</p>
                  </div>
                </div>

                <div className="flex items-center gap-2.5 sm:gap-3.5 flex-wrap justify-end">
                  <div className="text-left sm:text-right">
                    <div className={`text-xl sm:text-2xl font-black transition-colors duration-200 mono ${priceFlash}`}>
                      {state.price ? `$${Number(state.price).toFixed(2)}` : '--.--'}
                    </div>
                    <div className="text-[10px] text-slate-400 mono">
                      Spd: <b className={state.spread <= state.max_spread ? "text-emerald-300" : "text-rose-400"}>${Number(state.spread).toFixed(2)}</b> / ${state.max_spread}
                    </div>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => sendCommand('RELOAD_HISTORY')}
                      title="إعادة مزامنة الشموع التاريخية من MEXC"
                      className="apple-btn px-2.5 py-1.5 rounded-xl text-[11px] font-bold bg-white/5 hover:bg-white/10 text-amber-300 border border-white/10"
                    >
                      🔄 مزامنة
                    </button>
                    <button
                      onClick={() => sendCommand('TOGGLE_BOT')}
                      className={`apple-btn px-3 py-1.5 rounded-xl text-[11px] font-bold border ${
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
                        className="apple-btn px-2.5 py-1.5 rounded-xl text-[11px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40"
                      >
                        إغلاق!
                      </button>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 overflow-x-auto no-scrollbar pt-1.5 border-t border-white/[0.06]">
                <div className="vision-pill px-3 py-1.5 shrink-0 flex items-center gap-3 min-w-[175px]">
                  <div>
                    <div className="text-[9px] text-slate-400">الرصيد والملكية (Equity)</div>
                    <div className={`text-xs sm:text-sm font-extrabold mono ${state.equity >= state.balance ? 'text-emerald-400' : 'text-rose-400'}`}>
                      ${Number(state.equity).toLocaleString('en-US', { minimumFractionDigits: 2 })}
                    </div>
                  </div>
                  <EquitySparkline initialBalance={state.initial_balance} currentEquity={state.equity} tradeLog={state.trade_log} />
                </div>

                <div className="vision-pill px-2.5 py-1.5 shrink-0 min-w-[95px]">
                  <div className="text-[9px] text-slate-400">صافي الربح</div>
                  <div className={`text-xs sm:text-sm font-bold mono ${kpis.net_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {kpis.net_pnl >= 0 ? '+' : ''}${Number(kpis.net_pnl).toFixed(2)}
                  </div>
                </div>

                <div className="vision-pill px-2.5 py-1.5 shrink-0 min-w-[105px]">
                  <div className="text-[9px] text-slate-400">نسبة النجاح</div>
                  <div className="text-xs sm:text-sm font-bold text-amber-400 mono">
                    {kpis.win_rate}%
                  </div>
                </div>

                <div className="vision-pill px-2.5 py-1.5 shrink-0 min-w-[85px]">
                  <div className="text-[9px] text-slate-400">أقصى سحب</div>
                  <div className="text-xs sm:text-sm font-bold text-rose-400 mono">
                    {Number(kpis.max_drawdown_pct).toFixed(2)}%
                  </div>
                </div>
              </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5 flex-1">
              <div className="lg:col-span-8 flex flex-col gap-3.5">
                <div className="vision-card p-3.5 flex-1 flex flex-col relative">
                  <div className="flex items-center justify-between gap-2 pb-2.5 mb-2 border-b border-white/[0.06]">
                    <div className="flex items-center gap-2.5">
                      <span className="text-xs font-bold text-slate-200">شارت فيوتشر الذهب (XAU_USDT {state.tf_min}m)</span>
                      <span className="vision-pill px-2.5 py-0.5 text-[11px] mono text-amber-300 font-semibold">
                        ⏱ {formatCountdown(state.candle_rem_s || 0)}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 text-[11px]">
                      <button
                        onClick={() => setShowOB(!showOB)}
                        className={`px-2.5 py-1 rounded-lg border transition ${
                          showOB ? 'bg-amber-500/15 text-amber-300 border-amber-500/30' : 'bg-white/5 text-slate-400 border-white/5'
                        }`}
                      >
                        Order Blocks
                      </button>
                      <button
                        onClick={() => setShowVP(!showVP)}
                        className={`px-2.5 py-1 rounded-lg border transition ${
                          showVP ? 'bg-purple-500/15 text-purple-300 border-purple-500/30' : 'bg-white/5 text-slate-400 border-white/5'
                        }`}
                      >
                        POC
                      </button>
                    </div>
                  </div>

                  <ErrorBoundary label="TradingChart">
                  <TradingChart
                    candles={snapshotCandles}
                    latestCandle={latestCandle}
                    zones={state.discovered_zones}
                    openPos={pos}
                    poc={cog.poc}
                    vah={cog.vah}
                    val={cog.val}
                    orderBlocks={state.order_blocks}
                    swingHigh={cog.swing_high}
                    swingLow={cog.swing_low}
                    emaFastSeries={emaFastSeries}
                    emaSlowSeries={emaSlowSeries}
                    emaFastPoint={emaFastPoint}
                    emaSlowPoint={emaSlowPoint}
                    showOB={showOB}
                    showVP={showVP}
                    showEMA={showEMA}
                  />
                  </ErrorBoundary>
                </div>

                <div className="vision-card p-4">
                  {renderActivePosition()}
                </div>
              </div>

              <div className="lg:col-span-4 flex flex-col gap-3.5">
                {renderCortexPanel()}
                {renderControlPanel()}
                {renderNeuralStream()}
              </div>
            </div>

          </div>
        </div>
      );
    }

    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(<ErrorBoundary label="App"><App /></ErrorBoundary>);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_terminal():
    return HTMLResponse(content=HTML_UI)


# ---- توافق ZeroGPU: واجهة Gradio صغيرة بدالة @spaces.GPU حقيقية مربوطة بها ----
try:
    import gradio as gr
    import spaces

    @spaces.GPU
    def _gpu_ping(text: str = "") -> str:
        return "ok"

    _gpu_demo = gr.Interface(
        fn=_gpu_ping,
        inputs=gr.Textbox(label="ping"),
        outputs=gr.Textbox(label="pong"),
        api_name="ping",
    )
    app = gr.mount_gradio_app(app, _gpu_demo, path="/gradio")
    print("ZeroGPU: Gradio + @spaces.GPU mounted at /gradio")
except Exception as _e:
    print("ZeroGPU/Gradio mount skipped:", repr(_e))


if __name__ == "__main__":
    if not DASHBOARD_TOKEN:
        print("WARNING: DASHBOARD_TOKEN غير مضبوط — أي شخص يصل للمنفذ يستطيع التحكم بالبوت.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")), reload=False)
