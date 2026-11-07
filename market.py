"""
market.py — Data Adapter: ดึงข้อมูลจริงจาก Yahoo Finance (yfinance) และ Google News RSS
กฎ: ถ้าดึงไม่ได้ให้คืน None / ค่าว่าง ห้ามสร้างตัวเลขขึ้นเอง
งบการเงินคำนวณอัตราส่วนด้วยโค้ด (deterministic) แล้วค่อย fallback ไปค่าที่ provider คำนวณให้ พร้อมระบุแหล่งที่มา
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from config import (
    CACHE_TTL_FUNDAMENTAL, CACHE_TTL_NEWS, CACHE_TTL_PRICE, HISTORY_PERIOD,
    to_set_symbol, to_yahoo_symbol,
)

HTTP_TIMEOUT = 10


def _num(x: Any) -> float | None:
    """แปลงเป็น float; ค่าไม่ถูกต้องคืน None"""
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


# ======================================================================
# ราคา
# ======================================================================
@st.cache_data(ttl=CACHE_TTL_PRICE, show_spinner=False)
def fetch_history(symbol: str, period: str = HISTORY_PERIOD) -> pd.DataFrame | None:
    """OHLCV รายวัน (ราคาไม่ปรับปันผล เพื่อให้ระดับราคาตรงกับกระดานจริง)"""
    try:
        df = yf.Ticker(to_yahoo_symbol(symbol)).history(
            period=period, interval="1d", auto_adjust=False, actions=False
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[df["Volume"] > 0]  # ตัดวันหยุด/วันที่ไม่มีการซื้อขาย
    return df if len(df) >= 30 else None


# ======================================================================
# พื้นฐาน
# ======================================================================
def _row(df: pd.DataFrame | None, names: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            s = s.dropna()
            if not s.empty:
                return s.sort_index(ascending=False)  # ล่าสุดอยู่หน้า
    return None


def _ttm(s: pd.Series | None) -> float | None:
    if s is None or len(s) < 4:
        return None
    return float(s.iloc[:4].sum())


def _yoy(s: pd.Series | None) -> float | None:
    """เทียบไตรมาสล่าสุดกับไตรมาสเดียวกันปีก่อน"""
    if s is None or len(s) < 5:
        return None
    cur, prev = float(s.iloc[0]), float(s.iloc[4])
    if prev == 0:
        return None
    return (cur - prev) / abs(prev)


@st.cache_data(ttl=CACHE_TTL_FUNDAMENTAL, show_spinner=False)
def fetch_fundamentals(symbol: str, last_price: float | None) -> dict:
    """
    คืน dict ของอัตราส่วนพื้นฐาน + แหล่งที่มาของแต่ละค่า
    ค่า 'computed' = คำนวณเองจากงบการเงิน, 'provider' = ค่าที่ Yahoo คำนวณ
    """
    out: dict[str, Any] = {"symbol": to_set_symbol(symbol), "sources": {}, "errors": []}
    t = yf.Ticker(to_yahoo_symbol(symbol))

    try:
        info = t.info or {}
    except Exception as e:
        info = {}
        out["errors"].append(f"info: {e}")

    out["name"] = info.get("longName") or info.get("shortName") or out["symbol"]
    out["sector"] = info.get("sector")
    out["industry"] = info.get("industry")
    out["summary"] = info.get("longBusinessSummary")
    out["currency"] = info.get("currency") or "THB"
    out["financial_currency"] = info.get("financialCurrency") or out["currency"]
    out["shares"] = _num(info.get("sharesOutstanding"))
    out["fy_end_month"] = None
    if info.get("lastFiscalYearEnd"):
        try:
            out["fy_end_month"] = datetime.fromtimestamp(
                int(info["lastFiscalYearEnd"]), tz=timezone.utc
            ).month
        except Exception:
            pass

    price = last_price or _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    out["price"] = price
    out["market_cap"] = (price * out["shares"]) if (price and out["shares"]) else _num(info.get("marketCap"))

    # ---------- งบการเงินรายไตรมาส ----------
    try:
        q_is = t.quarterly_income_stmt
    except Exception:
        q_is = None
    try:
        q_bs = t.quarterly_balance_sheet
    except Exception:
        q_bs = None
    try:
        q_cf = t.quarterly_cashflow
    except Exception:
        q_cf = None

    revenue = _row(q_is, ["Total Revenue", "Operating Revenue"])
    net_income = _row(q_is, ["Net Income Common Stockholders", "Net Income",
                             "Net Income Continuous Operations"])
    equity = _row(q_bs, ["Common Stock Equity", "Stockholders Equity"])
    total_debt = _row(q_bs, ["Total Debt"])
    cur_assets = _row(q_bs, ["Current Assets"])
    cur_liab = _row(q_bs, ["Current Liabilities"])
    ocf = _row(q_cf, ["Operating Cash Flow"])
    capex = _row(q_cf, ["Capital Expenditure"])

    ni_ttm = _ttm(net_income)
    rev_ttm = _ttm(revenue)
    eq_latest = float(equity.iloc[0]) if equity is not None else None
    same_ccy = out["financial_currency"] == out["currency"]

    def put(key: str, computed: float | None, provider: Any, src_note: str = "งบการเงิน (คำนวณเอง)"):
        if computed is not None:
            out[key] = computed
            out["sources"][key] = src_note
        else:
            v = _num(provider)
            out[key] = v
            if v is not None:
                out["sources"][key] = "Yahoo Finance"

    # P/E, P/BV — คำนวณเองได้เฉพาะเมื่อสกุลเงินงบ = สกุลเงินราคา (เช่น PTTEP รายงานงบเป็น USD)
    mcap = out["market_cap"]
    pe_c = (mcap / ni_ttm) if (same_ccy and mcap and ni_ttm and ni_ttm > 0) else None
    pbv_c = (mcap / eq_latest) if (same_ccy and mcap and eq_latest and eq_latest > 0) else None
    put("pe", pe_c, info.get("trailingPE"))
    put("pbv", pbv_c, info.get("priceToBook"))
    out["forward_pe"] = _num(info.get("forwardPE"))

    roe_c = (ni_ttm / eq_latest) if (ni_ttm is not None and eq_latest and eq_latest > 0) else None
    put("roe", roe_c, info.get("returnOnEquity"))

    npm_c = (ni_ttm / rev_ttm) if (ni_ttm is not None and rev_ttm and rev_ttm > 0) else None
    put("net_margin", npm_c, info.get("profitMargins"))

    de_c = (float(total_debt.iloc[0]) / eq_latest) if (total_debt is not None and eq_latest and eq_latest > 0) else None
    de_prov = _num(info.get("debtToEquity"))
    put("debt_to_equity", de_c, (de_prov / 100) if de_prov is not None else None)  # Yahoo ให้เป็น %

    cr_c = (float(cur_assets.iloc[0]) / float(cur_liab.iloc[0])) if (
        cur_assets is not None and cur_liab is not None and float(cur_liab.iloc[0]) > 0) else None
    put("current_ratio", cr_c, info.get("currentRatio"))

    put("revenue_yoy", _yoy(revenue), info.get("revenueGrowth"), "งบไตรมาสล่าสุด YoY (คำนวณเอง)")
    put("earnings_yoy", _yoy(net_income), info.get("earningsGrowth"), "งบไตรมาสล่าสุด YoY (คำนวณเอง)")

    fcf_c = None
    if ocf is not None and len(ocf) >= 4:
        fcf_c = _ttm(ocf) + (_ttm(capex) or 0.0)  # capex เป็นค่าติดลบใน yfinance
    put("fcf_ttm", fcf_c, info.get("freeCashflow"))

    out["payout_ratio"] = _num(info.get("payoutRatio"))
    out["beta"] = _num(info.get("beta"))
    out["target_mean_price"] = _num(info.get("targetMeanPrice"))
    out["analyst_count"] = info.get("numberOfAnalystOpinions")
    if out["target_mean_price"] is not None:
        out["sources"]["target_mean_price"] = "Yahoo Finance (ค่าเฉลี่ยนักวิเคราะห์)"

    # ไตรมาสล่าสุดที่มีงบ (ใช้ประมาณวันประกาศงบถัดไป)
    out["last_reported_quarter"] = None
    if q_is is not None and not q_is.empty:
        try:
            out["last_reported_quarter"] = pd.to_datetime(max(q_is.columns)).date()
        except Exception:
            pass

    # ตารางกำไรรายไตรมาส (สำหรับแสดงผล)
    if revenue is not None or net_income is not None:
        tbl = pd.DataFrame({"รายได้": revenue, "กำไรสุทธิ": net_income}).sort_index(ascending=False).head(8)
        tbl.index = [pd.to_datetime(i).strftime("%Y-%m") for i in tbl.index]
        out["quarterly_table"] = tbl
    else:
        out["quarterly_table"] = None
    return out


# ======================================================================
# ปันผล / ปฏิทิน
# ======================================================================
@st.cache_data(ttl=CACHE_TTL_FUNDAMENTAL, show_spinner=False)
def fetch_dividends(symbol: str) -> pd.Series:
    """ประวัติปันผล: index = วัน XD, value = บาท/หุ้น"""
    try:
        s = yf.Ticker(to_yahoo_symbol(symbol)).dividends
    except Exception:
        return pd.Series(dtype=float)
    if s is None or s.empty:
        return pd.Series(dtype=float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s[s > 0].sort_index()


@st.cache_data(ttl=CACHE_TTL_FUNDAMENTAL, show_spinner=False)
def fetch_calendar(symbol: str) -> dict:
    """ปฏิทินที่ provider รายงาน (Earnings Date / Ex-Dividend Date / Dividend Date)"""
    try:
        cal = yf.Ticker(to_yahoo_symbol(symbol)).calendar
    except Exception:
        return {}
    if cal is None:
        return {}
    if isinstance(cal, pd.DataFrame):  # yfinance รุ่นเก่า
        cal = {k: (v.iloc[0] if len(v) else None) for k, v in cal.to_dict(orient="series").items()} \
            if not cal.empty else {}

    def to_dates(v) -> list:
        vals = v if isinstance(v, (list, tuple)) else [v]
        res = []
        for x in vals:
            try:
                d = pd.to_datetime(x)
                if pd.notna(d):
                    res.append(d.date())
            except Exception:
                pass
        return res

    return {
        "earnings": to_dates(cal.get("Earnings Date")),
        "ex_dividend": to_dates(cal.get("Ex-Dividend Date")),
        "dividend_pay": to_dates(cal.get("Dividend Date")),
    }


# ======================================================================
# ข่าว
# ======================================================================
def _yf_news(symbol: str) -> list[dict]:
    try:
        raw = yf.Ticker(to_yahoo_symbol(symbol)).news or []
    except Exception:
        return []
    items = []
    for n in raw[:15]:
        c = n.get("content", n)  # yfinance >=0.2.50 ห่อไว้ใน 'content'
        title = c.get("title")
        if not title:
            continue
        url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
        pub = c.get("pubDate") or c.get("providerPublishTime")
        try:
            dt = (datetime.fromtimestamp(pub, tz=timezone.utc) if isinstance(pub, (int, float))
                  else pd.to_datetime(pub, utc=True).to_pydatetime())
        except Exception:
            dt = None
        publisher = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) \
            else c.get("publisher")
        items.append({"title": title, "url": url, "published": dt, "source": publisher or "Yahoo Finance"})
    return items


def _google_news(query: str) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=th&gl=TH&ceid=TH:th"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        return []
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate"))
        except Exception:
            dt = None
        src = it.find("source")
        items.append({
            "title": title,
            "url": it.findtext("link"),
            "published": dt,
            "source": src.text if src is not None else "Google News",
        })
        if len(items) >= 15:
            break
    return items


@st.cache_data(ttl=CACHE_TTL_NEWS, show_spinner=False)
def fetch_news(symbol: str, max_items: int = 12) -> list[dict]:
    sym = to_set_symbol(symbol)
    items = _google_news(f"หุ้น {sym}") + _yf_news(sym)
    seen, uniq = set(), []
    for n in items:
        key = n["title"][:60].lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(n)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    uniq.sort(key=lambda n: n["published"] or epoch, reverse=True)
    return uniq[:max_items]
