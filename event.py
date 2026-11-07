"""
events.py — Future Event Timeline Engine + Dividend Outlook
ทุกเหตุการณ์มีสถานะความน่าเชื่อถือ:
  ✅ Confirmed  — provider รายงานวันที่ไว้ (ยังควรเช็กกับ set.or.th)
  📌 Expected   — กำหนดตามกฎหมาย/ประกาศ ตลท. (deadline ส่งงบ, ประชุมผู้ถือหุ้น)
  🔮 Estimated  — ประมาณจากรูปแบบในอดีต (เช่น วัน XD ปีก่อน) — ต้องรอประกาศจริง
  📰 News       — ข่าวที่เกิดขึ้นแล้ว
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from config import AGM_DEADLINE_MONTHS, ANNUAL_DEADLINE_DAYS, QUARTERLY_DEADLINE_DAYS

STATUS_ICON = {"Confirmed": "✅", "Expected": "📌", "Estimated": "🔮", "News": "📰", "Past": "🕘"}


def _quarter_end(d: date) -> date:
    q_month = ((d.month - 1) // 3 + 1) * 3
    nxt = date(d.year + (q_month == 12), (q_month % 12) + 1, 1)
    return nxt - timedelta(days=1)


def _add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y, m = d.year + m // 12, m % 12 + 1
    days = [31, 29 if (y % 4 == 0 and (y % 100 or y % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return date(y, m, min(d.day, days[m - 1]))


def _ev(d: date, kind: str, title: str, status: str, detail: str, source: str, today: date) -> dict:
    return {"date": d, "days_to": (d - today).days, "type": kind, "title": title,
            "status": status, "icon": STATUS_ICON[status], "detail": detail, "source": source}


# ======================================================================
# Dividend
# ======================================================================
def dividend_outlook(divs: pd.Series, price: float | None, calendar: dict, today: date | None = None) -> dict:
    today = today or date.today()
    out = {"has_dividend": not divs.empty, "history": None, "ttm_dps": None, "ttm_yield": None,
           "per_year": None, "next": [], "avg_yield_3y": None, "streak_years": 0}
    if divs.empty:
        return out

    hist = divs[divs.index >= pd.Timestamp(today) - pd.DateOffset(years=5)]
    out["history"] = pd.DataFrame({"วัน XD": [d.date() for d in hist.index],
                                   "ปันผล (บาท/หุ้น)": hist.values}).iloc[::-1].reset_index(drop=True)

    last_12m = divs[divs.index > pd.Timestamp(today) - pd.Timedelta(days=365)]
    out["ttm_dps"] = float(last_12m.sum()) if not last_12m.empty else 0.0
    if price:
        out["ttm_yield"] = out["ttm_dps"] / price * 100
        by_year = divs.groupby(divs.index.year).sum()
        full_years = by_year[[y for y in by_year.index if y < today.year]].tail(3)
        if len(full_years):
            out["avg_yield_3y"] = float(full_years.mean()) / price * 100

    years = sorted(set(divs.index.year))
    streak, y = 0, today.year - 1
    while y in years:
        streak, y = streak + 1, y - 1
    out["streak_years"] = streak
    recent_years = [yy for yy in years if today.year - 3 <= yy < today.year]
    if recent_years:
        out["per_year"] = round(sum((divs.index.year == yy).sum() for yy in recent_years) / len(recent_years), 1)

    # 1) วันที่ provider รายงาน (ยืนยันแล้ว)
    for d in calendar.get("ex_dividend", []):
        if d >= today:
            pay = [p for p in calendar.get("dividend_pay", []) if p >= d]
            out["next"].append({
                "xd": d, "status": "Confirmed", "amount": None,
                "pay": pay[0] if pay else None,
                "pay_note": "วันจ่ายจาก provider" if pay else "ยังไม่มีวันจ่าย — ปกติ 2–4 สัปดาห์หลัง XD",
                "basis": "Yahoo Finance calendar",
            })

    # 2) ประมาณจากรอบเดียวกันของปีที่แล้ว
    confirmed_months = {(n["xd"].year, n["xd"].month) for n in out["next"]}
    for d, amt in last_12m.items():
        est = (d + pd.DateOffset(years=1)).date()
        if est < today:
            continue
        if any(abs((est - n["xd"]).days) < 45 for n in out["next"]):
            continue  # มีวันยืนยันในช่วงนั้นแล้ว
        if (est.year, est.month) in confirmed_months:
            continue
        out["next"].append({
            "xd": est, "status": "Estimated", "amount": float(amt),
            "pay": None,
            "pay_note": f"ประมาณ {est + timedelta(days=14):%d/%m/%Y} – {est + timedelta(days=28):%d/%m/%Y} (2–4 สัปดาห์หลัง XD)",
            "basis": f"อ้างอิงรอบ XD {d:%d/%m/%Y} จ่าย {amt:.2f} บาท (ยอดจริงขึ้นกับกำไรและมติกรรมการ)",
        })
    out["next"].sort(key=lambda n: n["xd"])
    return out


# ======================================================================
# Timeline
# ======================================================================
def build_timeline(fund: dict, div_out: dict, calendar: dict, news: list[dict],
                   today: date | None = None, horizon_days: int = 180) -> list[dict]:
    today = today or date.today()
    events: list[dict] = []
    src_cal = "Yahoo Finance calendar"

    # ---------- ประกาศงบ ----------
    confirmed_earn = [d for d in calendar.get("earnings", []) if d >= today]
    for d in confirmed_earn[:1]:
        events.append(_ev(d, "earnings", "ประกาศงบการเงิน", "Confirmed",
                          "วันที่จาก provider — ตรวจสอบกับ set.or.th อีกครั้ง", src_cal, today))

    fy_end_month = fund.get("fy_end_month") or 12
    last_q = fund.get("last_reported_quarter")
    q = _quarter_end(last_q + timedelta(days=1)) if last_q else _quarter_end(today - timedelta(days=95))
    for _ in range(3):  # ไตรมาสถัด ๆ ไปที่ยังไม่ประกาศ
        is_annual = q.month == fy_end_month
        deadline = q + timedelta(days=ANNUAL_DEADLINE_DAYS if is_annual else QUARTERLY_DEADLINE_DAYS)
        if deadline >= today and not any(abs((deadline - d).days) < 30 for d in confirmed_earn):
            label = "งบปี" if is_annual else f"งบไตรมาส (สิ้นสุด {q:%d/%m/%Y})"
            events.append(_ev(deadline, "earnings", f"เส้นตายส่ง{label}", "Expected",
                              "ตามเกณฑ์ ตลท.: งบไตรมาส ≤45 วัน, งบปี ≤60 วันหลังสิ้นงวด — บริษัทอาจประกาศก่อน",
                              "เกณฑ์ ตลท./ก.ล.ต.", today))
        q = _quarter_end(q + timedelta(days=1))

    # ---------- ประชุมผู้ถือหุ้น ----------
    y = today.year if today.month > fy_end_month else today.year - 1
    fy_end = _add_months(date(y, fy_end_month, 1), 1) - timedelta(days=1)  # วันสุดท้ายของเดือนสิ้นปีบัญชี
    agm = _add_months(fy_end, AGM_DEADLINE_MONTHS)
    if agm < today:
        agm = _add_months(agm, 12)
    events.append(_ev(agm, "agm", "ประชุมสามัญผู้ถือหุ้น (AGM) — เส้นตาย", "Expected",
                      "ต้องจัดภายใน 4 เดือนหลังสิ้นปีบัญชี — อนุมัติปันผลงวดสุดท้ายในที่ประชุมนี้",
                      "พ.ร.บ.บริษัทมหาชนจำกัด", today))

    # ---------- ปันผล ----------
    for n in div_out.get("next", []):
        amt = f" (รอบเดียวกันปีก่อน {n['amount']:.2f} บาท)" if n.get("amount") else ""
        events.append(_ev(n["xd"], "dividend", f"วัน XD ปันผล{amt}", n["status"], n["basis"],
                          src_cal if n["status"] == "Confirmed" else "ประวัติปันผล", today))
        if n.get("pay"):
            events.append(_ev(n["pay"], "dividend", "วันจ่ายปันผล", n["status"], n["pay_note"], src_cal, today))

    events = [e for e in events if 0 <= e["days_to"] <= horizon_days]

    # ---------- ข่าวล่าสุด ----------
    for nw in news[:8]:
        if nw.get("published"):
            d = nw["published"].date() if isinstance(nw["published"], datetime) else nw["published"]
            events.append({**_ev(d, "news", nw["title"], "News", nw.get("source") or "", nw.get("source") or "", today),
                           "url": nw.get("url")})

    events.sort(key=lambda e: (e["status"] == "News", e["date"] if e["status"] != "News" else -e["date"].toordinal()))
    return events


def impact_hint(event: dict, fund: dict, div_out: dict, price: float | None) -> str:
    """คำอธิบายผลกระทบแบบกฎตายตัว (ไม่ใช่การเดา) ให้ผู้ใช้เตรียมตัว"""
    if event["type"] == "dividend" and "XD" in event["title"] and price:
        amt = None
        for n in div_out.get("next", []):
            if n["xd"] == event["date"]:
                amt = n.get("amount")
        if amt:
            return (f"ต้องถือหุ้นก่อนวัน XD | ในวัน XD ราคามักเปิดลดลงใกล้เคียงยอดปันผล (~{amt / price * 100:.1f}%) "
                    "| เงินปันผลหักภาษี ณ ที่จ่าย 10%")
        return "ต้องถือหุ้นก่อนวัน XD | ราคามักปรับลดตามยอดปันผลในวัน XD | หักภาษี ณ ที่จ่าย 10%"
    if event["type"] == "earnings":
        ey = fund.get("earnings_yoy")
        trend = "" if ey is None else f" | งบล่าสุดกำไร YoY {ey * 100:+.0f}% — ตลาดจะเทียบกับตัวเลขนี้"
        return "ความผันผวนมักสูงช่วงประกาศงบ พิจารณาลดขนาดไม้หรือรอดูงบก่อนเข้าซื้อเพิ่ม" + trend
    if event["type"] == "agm":
        return "ที่ประชุมอนุมัติปันผลงวดสุดท้าย/แผนธุรกิจ — ติดตามวาระการประชุมบน set.or.th"
    return ""
