"""
analysis.py — Financial Calculation Engine (deterministic, pandas/numpy เท่านั้น)
- อินดิเคเตอร์: EMA, RSI(Wilder), MACD, ATR(Wilder), Volume MA
- Price Action: แนวรับ/แนวต้านจาก pivot, โครงสร้าง Higher-High/Higher-Low, Breakout, รูปแบบแท่งเทียน
- คะแนนเทคนิค และคะแนนพื้นฐาน (0-100) — ค่าที่ไม่มีข้อมูลจะไม่ถูกนับ และน้ำหนักถูกปรับสัดส่วนใหม่
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import PIVOT_WINDOW, SR_LOOKBACK_DAYS


# ======================================================================
# Indicators
# ======================================================================
def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    for n in (20, 50, 200):
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()

    delta = c.diff()
    gain = _wilder(delta.clip(lower=0), 14)
    loss = _wilder(-delta.clip(upper=0), 14)
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = (100 - 100 / (1 + rs)).where(loss != 0, 100.0)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    prev_close = c.shift(1)
    tr = pd.concat([d["High"] - d["Low"],
                    (d["High"] - prev_close).abs(),
                    (d["Low"] - prev_close).abs()], axis=1).max(axis=1)
    d["ATR"] = _wilder(tr, 14)
    d["VOL_MA20"] = d["Volume"].rolling(20).mean()
    d["HIGH20"] = d["High"].rolling(20).max()
    d["VALUE"] = d["Close"] * d["Volume"]  # มูลค่าซื้อขาย (บาท)
    return d


# ======================================================================
# Support / Resistance (Price Action)
# ======================================================================
def find_pivots(df: pd.DataFrame, window: int = PIVOT_WINDOW) -> tuple[list, list]:
    """pivot high/low แบบ fractal: จุดสูง/ต่ำสุดในหน้าต่าง ±window แท่ง"""
    h, l = df["High"].values, df["Low"].values
    highs, lows = [], []
    for i in range(window, len(df) - window):
        if h[i] == h[i - window:i + window + 1].max():
            highs.append((df.index[i], float(h[i])))
        if l[i] == l[i - window:i + window + 1].min():
            lows.append((df.index[i], float(l[i])))
    return highs, lows


def _cluster(levels: list[float], tol: float) -> list[dict]:
    """รวมระดับราคาที่ใกล้กันภายใน tol ให้เป็นโซนเดียว นับจำนวนครั้งที่ทดสอบ"""
    if not levels:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for p in levels[1:]:
        if p - np.mean(clusters[-1]) <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [{"price": float(np.mean(c)), "touches": len(c)} for c in clusters]


def support_resistance(df: pd.DataFrame) -> dict:
    d = df.tail(SR_LOOKBACK_DAYS)
    price = float(d["Close"].iloc[-1])
    atr = float(d["ATR"].iloc[-1]) if "ATR" in d and pd.notna(d["ATR"].iloc[-1]) else price * 0.02
    highs, lows = find_pivots(d)
    zones = _cluster([p for _, p in highs] + [p for _, p in lows], tol=0.5 * atr)
    supports = sorted([z for z in zones if z["price"] < price], key=lambda z: -z["price"])
    resistances = sorted([z for z in zones if z["price"] > price], key=lambda z: z["price"])
    return {
        "supports": supports[:4],
        "resistances": resistances[:4],
        "pivot_highs": highs,
        "pivot_lows": lows,
        "high_52w": float(df["High"].tail(250).max()),
        "low_52w": float(df["Low"].tail(250).min()),
    }


def market_structure(pivot_highs: list, pivot_lows: list) -> str:
    """เทียบ swing 2 จุดล่าสุด: HH+HL = ขาขึ้น, LH+LL = ขาลง"""
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return "ไม่ชัดเจน"
    hh = pivot_highs[-1][1] > pivot_highs[-2][1]
    hl = pivot_lows[-1][1] > pivot_lows[-2][1]
    if hh and hl:
        return "ขาขึ้น (Higher High + Higher Low)"
    if not hh and not hl:
        return "ขาลง (Lower High + Lower Low)"
    return "ไซด์เวย์ / กำลังเปลี่ยนแนวโน้ม"


def candle_patterns(df: pd.DataFrame) -> list[str]:
    """รูปแบบแท่งเทียนกลับตัวฝั่งบวกของแท่งล่าสุด"""
    out = []
    if len(df) < 3:
        return out
    o, h, l, c = (df[k].iloc[-1] for k in ("Open", "High", "Low", "Close"))
    po, pc = df["Open"].iloc[-2], df["Close"].iloc[-2]
    body, rng = abs(c - o), h - l
    if rng > 0:
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        if lower_wick >= 2 * body and upper_wick <= body and body / rng < 0.35:
            out.append("Hammer / Pin bar (ไส้ล่างยาว — มีแรงซื้อรับ)")
        if upper_wick >= 2 * body and lower_wick <= body and body / rng < 0.35:
            out.append("Shooting star (ไส้บนยาว — มีแรงขายกด)")
    if pc < po and c > o and c >= po and o <= pc:
        out.append("Bullish Engulfing (กลืนกินขาขึ้น)")
    if pc > po and c < o and c <= po and o >= pc:
        out.append("Bearish Engulfing (กลืนกินขาลง)")
    return out


# ======================================================================
# Technical summary + score
# ======================================================================
def technical_snapshot(df: pd.DataFrame) -> dict:
    d = df
    last = d.iloc[-1]
    prev = d.iloc[-2]
    price = float(last["Close"])
    sr = support_resistance(d)
    snap = {
        "price": price,
        "date": d.index[-1].date(),
        "change_pct": (price / float(prev["Close"]) - 1) * 100,
        "ema20": _f(last.get("EMA20")), "ema50": _f(last.get("EMA50")), "ema200": _f(last.get("EMA200")),
        "rsi": _f(last.get("RSI")),
        "macd": _f(last.get("MACD")), "macd_signal": _f(last.get("MACD_signal")),
        "macd_hist": _f(last.get("MACD_hist")), "macd_hist_prev": _f(prev.get("MACD_hist")),
        "atr": _f(last.get("ATR")),
        "atr_pct": (_f(last.get("ATR")) or 0) / price * 100,
        "vol_ratio": (float(last["Volume"]) / float(last["VOL_MA20"])) if pd.notna(last.get("VOL_MA20")) and last["VOL_MA20"] > 0 else None,
        "avg_value_20d": float(d["VALUE"].tail(20).mean()),
        "ret_1m": _ret(d, 21), "ret_3m": _ret(d, 63), "ret_6m": _ret(d, 126),
        "structure": market_structure(sr["pivot_highs"], sr["pivot_lows"]),
        "patterns": candle_patterns(d),
        "breakout_20d": bool(price >= float(d["HIGH20"].iloc[-2])) if pd.notna(d["HIGH20"].iloc[-2]) else False,
        **{k: sr[k] for k in ("supports", "resistances", "high_52w", "low_52w")},
    }
    snap["dist_52w_high_pct"] = (price / snap["high_52w"] - 1) * 100
    return snap


def technical_score(s: dict) -> tuple[float | None, list[str]]:
    """คะแนนเทคนิค 0-100 พร้อมเหตุผล (ทุกเกณฑ์เป็นกฎตายตัว)"""
    pts, maxpts, why = 0.0, 0.0, []

    def add(cond_pts: float, full: float, text: str | None):
        nonlocal pts, maxpts
        pts += cond_pts
        maxpts += full
        if text:
            why.append(text)

    p = s["price"]
    if s["ema200"]:
        add(20 if p > s["ema200"] else 0, 20,
            "ราคาอยู่เหนือ EMA200 (แนวโน้มหลักขาขึ้น)" if p > s["ema200"] else "ราคาต่ำกว่า EMA200 (แนวโน้มหลักอ่อนแอ)")
    if s["ema50"] and s["ema20"]:
        aligned = s["ema20"] > s["ema50"] and p > s["ema20"]
        add(15 if aligned else (7 if p > s["ema50"] else 0), 15,
            "EMA20 > EMA50 และราคาอยู่เหนือ EMA20" if aligned else None)
    if s["rsi"] is not None:
        r = s["rsi"]
        rp = 15 if 50 <= r <= 68 else 10 if 40 <= r < 50 else 6 if 68 < r <= 75 else 2
        add(rp, 15, f"RSI {r:.0f}" + (" (โมเมนตัมดี ไม่ร้อนแรงเกิน)" if 50 <= r <= 68 else
                                      " (Overbought ระวังแรงขายทำกำไร)" if r > 70 else
                                      " (อ่อนแรง)" if r < 40 else ""))
    if s["macd_hist"] is not None and s["macd_hist_prev"] is not None:
        rising = s["macd_hist"] > s["macd_hist_prev"]
        add(15 if (s["macd_hist"] > 0 and rising) else 8 if rising else 3 if s["macd_hist"] > 0 else 0, 15,
            "MACD histogram เป็นบวกและเพิ่มขึ้น" if (s["macd_hist"] > 0 and rising)
            else "MACD กำลังฟื้นตัว" if rising else None)
    st_ = s["structure"]
    add(15 if st_.startswith("ขาขึ้น") else 7 if st_.startswith("ไซด์") else 0, 15, f"โครงสร้างราคา: {st_}")
    if s["vol_ratio"] is not None:
        vr = s["vol_ratio"]
        good = vr >= 1.5 and s["change_pct"] > 0
        add(10 if good else 5 if vr >= 1 else 2, 10,
            f"Volume {vr:.1f} เท่าค่าเฉลี่ย 20 วัน ขณะราคาบวก" if good else None)
    if s["ret_3m"] is not None:
        add(10 if s["ret_3m"] > 5 else 5 if s["ret_3m"] > -5 else 0, 10, f"ผลตอบแทน 3 เดือน {s['ret_3m']:+.1f}%")
    score = round(pts / maxpts * 100, 1) if maxpts else None
    return score, why


# ======================================================================
# Fundamental score (Quality)
# ======================================================================
def fundamental_score(f: dict) -> tuple[float | None, list[str], float]:
    """
    คะแนนพื้นฐาน 0-100 + เหตุผล + data coverage (สัดส่วนน้ำหนักที่มีข้อมูลจริง)
    น้ำหนัก: ความสามารถทำกำไร 25, สุขภาพการเงิน 25, การเติบโต 20, กระแสเงินสด 15, มูลค่า 15
    """
    rules = []  # (weight, score0to1 or None, text)

    roe = f.get("roe")
    rules.append((15, None if roe is None else min(max(roe / 0.15, 0), 1),
                  None if roe is None else f"ROE {roe*100:.1f}%" + (" (ดีมาก ≥15%)" if roe >= 0.15 else " (ต่ำ <8%)" if roe < 0.08 else "")))
    nm = f.get("net_margin")
    rules.append((10, None if nm is None else min(max(nm / 0.12, 0), 1),
                  None if nm is None else f"อัตรากำไรสุทธิ {nm*100:.1f}%"))

    de = f.get("debt_to_equity")
    rules.append((15, None if de is None else (1.0 if de <= 0.5 else 0.7 if de <= 1 else 0.4 if de <= 2 else 0.1),
                  None if de is None else f"หนี้สินที่มีภาระดอกเบี้ย/ทุน (D/E) {de:.2f} เท่า"))
    cr = f.get("current_ratio")
    rules.append((10, None if cr is None else (1.0 if cr >= 1.5 else 0.6 if cr >= 1 else 0.2),
                  None if cr is None else f"Current ratio {cr:.2f} เท่า"))

    ry = f.get("revenue_yoy")
    rules.append((8, None if ry is None else min(max((ry + 0.05) / 0.20, 0), 1),
                  None if ry is None else f"รายได้ไตรมาสล่าสุด YoY {ry*100:+.1f}%"))
    ey = f.get("earnings_yoy")
    rules.append((12, None if ey is None else min(max((ey + 0.10) / 0.35, 0), 1),
                  None if ey is None else f"กำไรสุทธิไตรมาสล่าสุด YoY {ey*100:+.1f}%"))

    fcf = f.get("fcf_ttm")
    rules.append((15, None if fcf is None else (1.0 if fcf > 0 else 0.0),
                  None if fcf is None else ("กระแสเงินสดอิสระ (FCF) 12 เดือนเป็นบวก" if fcf > 0 else "FCF 12 เดือนติดลบ")))

    pe = f.get("pe")
    rules.append((10, None if pe is None or pe <= 0 else (1.0 if pe <= 12 else 0.7 if pe <= 18 else 0.4 if pe <= 25 else 0.1),
                  None if pe is None else f"P/E {pe:.1f} เท่า"))
    pbv = f.get("pbv")
    rules.append((5, None if pbv is None or pbv <= 0 else (1.0 if pbv <= 1.2 else 0.6 if pbv <= 2.5 else 0.3),
                  None if pbv is None else f"P/BV {pbv:.2f} เท่า"))

    total_w = sum(w for w, _, _ in rules)
    have = [(w, s, t) for w, s, t in rules if s is not None]
    used_w = sum(w for w, _, _ in have)
    if used_w == 0:
        return None, [], 0.0
    score = sum(w * s for w, s, _ in have) / used_w * 100
    return round(score, 1), [t for _, _, t in have if t], used_w / total_w


def _f(x) -> float | None:
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _ret(df: pd.DataFrame, n: int) -> float | None:
    if len(df) <= n:
        return None
    return (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-1 - n]) - 1) * 100
