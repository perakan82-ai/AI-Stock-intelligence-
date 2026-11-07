"""
app.py — Thai Stock AI Intelligence Lab (Streamlit)
รัน:  streamlit run app.py
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ai
from analysis import add_indicators, fundamental_score, technical_score, technical_snapshot
from config import (
    APP_TITLE, APP_VERSION, DATA_SOURCE_NAME, DEFAULT_SCAN_LIST, DEFAULT_WATCHLIST, DISCLAIMER,
    STATUS_COLORS, to_set_symbol,
)
from events import build_timeline, dividend_outlook, impact_hint
from market import fetch_calendar, fetch_dividends, fetch_fundamentals, fetch_history, fetch_news
from signals import build_trade_plan, combined_score

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")


# ======================================================================
# Helpers: format (None → "ไม่มีข้อมูล" เสมอ ห้ามเติมค่าเอง)
# ======================================================================
NA = "—"


def f_num(x, d=2):
    return NA if x is None else f"{x:,.{d}f}"


def f_ratio_pct(x, d=1):
    return NA if x is None else f"{x * 100:.{d}f}%"


def f_money(x):
    if x is None:
        return NA
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:,.1f} พันล้าน"
    if a >= 1e6:
        return f"{x / 1e6:,.1f} ล้าน"
    return f"{x:,.0f}"


def parse_symbols(text: str) -> list[str]:
    seen, out = set(), []
    for s in text.replace("\n", ",").split(","):
        s = to_set_symbol(s)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ======================================================================
# Core pipeline (ทุกตัวเลขมาจากโค้ด)
# ======================================================================
def analyze_symbol(sym: str, capital: float, risk_pct: float) -> dict | None:
    hist = fetch_history(sym)
    if hist is None:
        return None
    df = add_indicators(hist)
    snap = technical_snapshot(df)
    tech, tech_why = technical_score(snap)
    fund = fetch_fundamentals(sym, snap["price"])
    fscore, fund_why, coverage = fundamental_score(fund)
    divs = fetch_dividends(sym)
    cal = fetch_calendar(sym)
    div_out = dividend_outlook(divs, snap["price"], cal)
    plan = build_trade_plan(snap, capital, risk_pct)
    comb = combined_score(tech, fscore, plan["primary"]["rr1"])
    return {
        "symbol": sym, "df": df, "snap": snap, "fund": fund, "div": div_out, "cal": cal, "plan": plan,
        "scores": {"combined": comb["score"], "complete": comb["complete"], "technical": tech,
                   "fundamental": fscore, "fund_coverage": coverage},
        "tech_why": tech_why, "fund_why": fund_why,
    }


def summary_row(r: dict) -> dict:
    p, s, d = r["plan"]["primary"], r["snap"], r["div"]
    nxt = d["next"][0] if d["next"] else None
    return {
        "หุ้น": r["symbol"],
        "สถานะ": f"{STATUS_COLORS.get(r['plan']['status'], '')} {r['plan']['status']}",
        "คะแนนรวม": r["scores"]["combined"],
        "เทคนิค": r["scores"]["technical"],
        "พื้นฐาน": r["scores"]["fundamental"],
        "ราคา": s["price"],
        "เปลี่ยน%": round(s["change_pct"], 2),
        "โซนซื้อ": f"{p['entry_low']:.2f}–{p['entry_high']:.2f}",
        "ตัดขาดทุน": p["stop"],
        "TP1": p["targets"][0]["price"] if p["targets"] else None,
        "R:R TP1": p["rr1"],
        "P/E": r["fund"].get("pe"),
        "Yield TTM%": round(d["ttm_yield"], 2) if d["ttm_yield"] is not None else None,
        "XD ถัดไป": f"{nxt['xd']:%d/%m/%y} ({'ยืนยัน' if nxt['status'] == 'Confirmed' else 'ประมาณ'})" if nxt else NA,
        "มูลค่าซื้อขาย/วัน (ลบ.)": round(s["avg_value_20d"] / 1e6, 1),
    }


# ======================================================================
# Chart
# ======================================================================
def price_chart(r: dict, days: int) -> go.Figure:
    df = r["df"].tail(days)
    p = r["plan"]["primary"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name="ราคา", increasing_line_color="#16a34a", decreasing_line_color="#dc2626"),
                  row=1, col=1)
    for col, color in (("EMA20", "#f59e0b"), ("EMA50", "#3b82f6"), ("EMA200", "#a855f7")):
        if df[col].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1.3, color=color)), row=1, col=1)
    # โซนซื้อ / ตัดขาดทุน / เป้าขาย
    fig.add_hrect(y0=p["entry_low"], y1=p["entry_high"], fillcolor="#22c55e", opacity=0.15, line_width=0,
                  annotation_text="โซนซื้อ", annotation_position="left", row=1, col=1)
    fig.add_hline(y=p["stop"], line_color="#ef4444", line_dash="dash", annotation_text=f"Stop {p['stop']:.2f}",
                  row=1, col=1)
    for i, t in enumerate(p["targets"], 1):
        fig.add_hline(y=t["price"], line_color="#0ea5e9", line_dash="dot",
                      annotation_text=f"TP{i} {t['price']:.2f}", row=1, col=1)
    for z in r["snap"]["supports"][:3]:
        fig.add_hline(y=z["price"], line_color="rgba(34,197,94,0.45)", line_width=1, row=1, col=1)
    for z in r["snap"]["resistances"][:3]:
        fig.add_hline(y=z["price"], line_color="rgba(239,68,68,0.45)", line_width=1, row=1, col=1)
    colors = ["#16a34a" if c >= o else "#dc2626" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume", showlegend=False),
                  row=2, col=1)
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.04), template="plotly_dark")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


# ======================================================================
# Detail page
# ======================================================================
def render_detail(r: dict, api_key: str):
    s, f, plan, d, sc = r["snap"], r["fund"], r["plan"], r["div"], r["scores"]
    sym = r["symbol"]
    st.subheader(f"{sym} — {f.get('name') or ''}")
    st.caption(f"{f.get('sector') or NA} · {f.get('industry') or NA} · ข้อมูลราคาถึง {s['date']:%d/%m/%Y} · "
               f"แหล่งข้อมูล: {DATA_SOURCE_NAME}")

    c = st.columns(6)
    c[0].metric("ราคาล่าสุด", f_num(s["price"]), f"{s['change_pct']:+.2f}%")
    c[1].metric("สถานะระบบ", f"{STATUS_COLORS.get(plan['status'], '')} {plan['status'].split(' (')[0]}")
    c[2].metric("คะแนนรวม", f_num(sc["combined"], 0))
    c[3].metric("เทคนิค", f_num(sc["technical"], 0))
    c[4].metric("พื้นฐาน", f_num(sc["fundamental"], 0), f"ข้อมูลครบ {sc['fund_coverage'] * 100:.0f}%",
                delta_color="off")
    c[5].metric("Yield 12 เดือน", NA if d["ttm_yield"] is None else f"{d['ttm_yield']:.2f}%")
    for why in plan["reasons"]:
        st.markdown(f"- {why}")

    st.plotly_chart(price_chart(r, st.session_state.get("chart_days", 180)), use_container_width=True)

    # ---------- จุดเข้า / จุดออก ----------
    st.markdown("### 🎯 แผนเทรด: จุดเข้า – จุดออก")
    plans = [plan["primary"]] + [x for x in (plan["pullback"], plan["breakout"]) if x and x is not plan["primary"]]
    cols = st.columns(len(plans))
    for col, p in zip(cols, plans):
        with col:
            tag = "⭐ แผนหลัก" if p is plan["primary"] else "แผนสำรอง"
            st.markdown(f"**{tag}: {p['name']}**")
            st.markdown(f"- **โซนเข้าซื้อ:** {p['entry_low']:.2f} – {p['entry_high']:.2f} บาท")
            st.markdown(f"- **อ้างอิงจาก:** {p['basis']}")
            st.markdown(f"- **ตัดขาดทุน:** {p['stop']:.2f} บาท ({p['stop_pct']:.1f}%)")
            if p["targets"]:
                st.dataframe(pd.DataFrame([{
                    "เป้า": f"TP{i}", "ราคา": t["price"], "กำไร%": round(t["pct"], 1),
                    "R:R": t["rr"], "ที่มา": t["basis"]} for i, t in enumerate(p["targets"], 1)]),
                    hide_index=True, use_container_width=True)
    sz = plan["sizing"]
    if sz:
        st.info(f"📐 **ขนาดไม้ที่แนะนำ:** {sz['shares']:,} หุ้น ({sz['lots']} board lot) ≈ {sz['cost']:,.0f} บาท · "
                f"ขาดทุนสูงสุดถ้าโดน Stop ≈ {sz['max_loss']:,.0f} บาท "
                f"(งบความเสี่ยง {sz['risk_amount']:,.0f} บาท){' · จำกัดด้วยเงินทุน' if sz['capped_by_cash'] else ''}")
    with st.expander("📤 กฎการขาย/ออกจากสถานะ", expanded=True):
        for rule in plan["exit_rules"]:
            st.markdown(f"- {rule}")

    # ---------- เทคนิค ----------
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### 📈 กราฟ & Price Action")
        st.markdown(f"- โครงสร้างราคา: **{s['structure']}**")
        st.markdown(f"- EMA20 / 50 / 200: {f_num(s['ema20'])} / {f_num(s['ema50'])} / {f_num(s['ema200'])}")
        st.markdown(f"- RSI(14): {f_num(s['rsi'], 1)} · ATR: {f_num(s['atr'])} ({s['atr_pct']:.1f}%/วัน)")
        sup_txt = ", ".join("%.2f" % z["price"] for z in s["supports"]) or NA
        res_txt = ", ".join("%.2f" % z["price"] for z in s["resistances"]) or NA
        st.markdown(f"- แนวรับ: {sup_txt}")
        st.markdown(f"- แนวต้าน: {res_txt}")
        st.markdown(f"- 52 สัปดาห์: {s['low_52w']:.2f} – {s['high_52w']:.2f} (ห่างจุดสูงสุด {s['dist_52w_high_pct']:.1f}%)")
        for ptn in s["patterns"]:
            st.markdown(f"- 🕯️ {ptn}")
        for w in r["tech_why"]:
            st.caption(f"• {w}")
    with t2:
        st.markdown("### 🏢 พื้นฐานบริษัท")
        rows = [
            ("P/E", f_num(f.get("pe"), 1), "pe"), ("Forward P/E", f_num(f.get("forward_pe"), 1), None),
            ("P/BV", f_num(f.get("pbv")), "pbv"), ("ROE", f_ratio_pct(f.get("roe")), "roe"),
            ("อัตรากำไรสุทธิ", f_ratio_pct(f.get("net_margin")), "net_margin"),
            ("D/E (หนี้ที่มีดอกเบี้ย)", f_num(f.get("debt_to_equity")), "debt_to_equity"),
            ("Current ratio", f_num(f.get("current_ratio")), "current_ratio"),
            ("รายได้ YoY", NA if f.get("revenue_yoy") is None else f"{f['revenue_yoy'] * 100:+.1f}%", "revenue_yoy"),
            ("กำไร YoY", NA if f.get("earnings_yoy") is None else f"{f['earnings_yoy'] * 100:+.1f}%", "earnings_yoy"),
            ("FCF 12 เดือน", f_money(f.get("fcf_ttm")), "fcf_ttm"),
            ("มูลค่าตลาด", f_money(f.get("market_cap")), None),
            ("ราคาเป้าเฉลี่ยนักวิเคราะห์", f_num(f.get("target_mean_price")) +
             (f" ({f['analyst_count']} ราย)" if f.get("analyst_count") else ""), "target_mean_price"),
        ]
        st.dataframe(pd.DataFrame([{"รายการ": a, "ค่า": b, "ที่มา": f["sources"].get(k, NA) if k else "Yahoo Finance"}
                                   for a, b, k in rows]), hide_index=True, use_container_width=True)
        if f.get("financial_currency") != f.get("currency"):
            st.caption(f"⚠️ งบการเงินรายงานเป็น {f.get('financial_currency')} — P/E, P/BV ใช้ค่าจาก Yahoo")
        for w in r["fund_why"]:
            st.caption(f"• {w}")
    if f.get("quarterly_table") is not None:
        with st.expander("งบกำไรขาดทุนรายไตรมาส"):
            st.dataframe(f["quarterly_table"].map(lambda v: f_money(v) if pd.notna(v) else NA),
                         use_container_width=True)
    if f.get("summary"):
        with st.expander("ลักษณะธุรกิจ"):
            st.write(f["summary"])

    # ---------- ปันผล ----------
    st.markdown("### 💰 ปันผล")
    if not d["has_dividend"]:
        st.warning("ไม่พบประวัติการจ่ายปันผลในแหล่งข้อมูล")
    else:
        c = st.columns(5)
        c[0].metric("ปันผล 12 เดือน", f"{d['ttm_dps']:.2f} บาท")
        c[1].metric("Yield 12 เดือน", NA if d["ttm_yield"] is None else f"{d['ttm_yield']:.2f}%")
        c[2].metric("Yield เฉลี่ย 3 ปี*", NA if d["avg_yield_3y"] is None else f"{d['avg_yield_3y']:.2f}%")
        c[3].metric("จ่ายปีละ", NA if d["per_year"] is None else f"{d['per_year']:g} ครั้ง")
        c[4].metric("จ่ายต่อเนื่อง", f"{d['streak_years']} ปี")
        st.caption("*คำนวณจากปันผลรวมรายปีเทียบราคาปัจจุบัน")
        if d["next"]:
            st.markdown("**📅 กำหนดปันผลถัดไป**")
            st.dataframe(pd.DataFrame([{
                "สถานะ": "✅ ยืนยันแล้ว" if n["status"] == "Confirmed" else "🔮 ประมาณการ",
                "วัน XD": f"{n['xd']:%d/%m/%Y}",
                "อีก (วัน)": (n["xd"] - date.today()).days,
                "ซื้อได้ไม่เกิน": "วันทำการก่อนวัน XD",
                "วันจ่าย": f"{n['pay']:%d/%m/%Y}" if n.get("pay") else n["pay_note"],
                "ยอดอ้างอิง": f"{n['amount']:.2f} บาท (รอบปีก่อน)" if n.get("amount") else "รอประกาศ",
                "ที่มา": n["basis"]} for n in d["next"]]), hide_index=True, use_container_width=True)
        else:
            st.caption("ยังไม่พบกำหนด XD ในช่วง 12 เดือนข้างหน้า")
        with st.expander("ประวัติปันผล 5 ปี"):
            st.dataframe(d["history"], hide_index=True, use_container_width=True)

    # ---------- Timeline ----------
    st.markdown("### 🗓️ Timeline เหตุการณ์ & ข่าว")
    news = fetch_news(sym)
    timeline = build_timeline(f, d, r["cal"], news)
    r["timeline"], r["news"] = timeline, news
    st.caption("✅ ยืนยันจาก provider · 📌 ตามเกณฑ์กฎหมาย/ตลท. · 🔮 ประมาณจากอดีต · 📰 ข่าวล่าสุด")
    for e in timeline:
        if e["status"] == "News":
            continue
        hint = impact_hint(e, f, d, s["price"])
        st.markdown(f"{e['icon']} **{e['date']:%d/%m/%Y}** (อีก {e['days_to']} วัน) — **{e['title']}**  \n"
                    f"<small>{e['detail']}{' · ' + hint if hint else ''}</small>", unsafe_allow_html=True)
    if news:
        with st.expander(f"📰 ข่าวล่าสุด ({len(news)})", expanded=True):
            for n in news:
                when = n["published"].strftime("%d/%m/%Y") if n.get("published") else NA
                st.markdown(f"- {when} · [{n['title']}]({n['url']}) — *{n.get('source') or ''}*")
    else:
        st.caption("ไม่พบข่าว")

    # ---------- AI ----------
    st.markdown("### 🤖 AI Analyst Team: Bull vs Bear")
    key = f"ai_{sym}"
    if st.button("ให้ AI วิเคราะห์หุ้นตัวนี้", key=f"btn_{sym}", disabled=not api_key):
        with st.spinner("AI 6 ตำแหน่งกำลังถกเถียง..."):
            payload = ai.build_payload(sym, f, s, plan, sc, d, timeline, news)
            st.session_state[key] = ai.analyze_stock(api_key, payload)
    if not api_key:
        st.caption("ใส่ Gemini API Key ที่แถบด้านซ้ายเพื่อใช้ AI")
    res = st.session_state.get(key)
    if res:
        if not res.get("ok"):
            st.error(f"AI ใช้งานไม่ได้: {res.get('error')}")
        else:
            st.success(res.get("summary", ""))
            ns = res.get("news_sentiment") or {}
            st.markdown(f"**Sentiment ข่าวรวม:** {ns.get('overall', NA)}")
            a, b = st.columns(2)
            with a:
                st.markdown("**🐂 ฝั่งกระทิง**")
                for x in res.get("bull_case", []):
                    st.markdown(f"- {x}")
                st.markdown("**🚀 Catalyst ที่ต้องติดตาม**")
                for x in res.get("catalysts", []):
                    st.markdown(f"- {x}")
            with b:
                st.markdown("**🐻 ฝั่งหมี**")
                for x in res.get("bear_case", []):
                    st.markdown(f"- {x}")
                st.markdown("**❌ เงื่อนไขที่ Thesis ผิด**")
                for x in res.get("thesis_invalidation", []):
                    st.markdown(f"- {x}")
            st.markdown(f"**มุมมองพื้นฐาน:** {res.get('fundamental_view', NA)}")
            st.markdown(f"**มุมมองเทคนิค:** {res.get('technical_view', NA)}")
            st.markdown(f"**กลยุทธ์รับมือเหตุการณ์ใกล้ถึง:** {res.get('event_strategy', NA)}")
            cv = res.get("chief_verdict") or {}
            st.markdown(f"**🧠 Chief Investment Analyst** "
                        f"({'เห็นด้วย' if cv.get('agree_with_system_status') else 'ไม่เห็นด้วย'}กับสถานะระบบ): "
                        f"{cv.get('comment', NA)}")
            items = ns.get("items") or []
            if items:
                with st.expander("Sentiment รายข่าว"):
                    st.dataframe(pd.DataFrame(items), hide_index=True, use_container_width=True)
            st.caption("ข้อความจาก AI เป็นความเห็นเชิงคุณภาพ ตัวเลขทั้งหมดมาจากระบบคำนวณ")


# ======================================================================
# Sidebar
# ======================================================================
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    secret_key = ""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    api_key = st.text_input("Gemini API Key", value=secret_key, type="password")
    watch_text = st.text_input("Watchlist (คั่นด้วย ,)", DEFAULT_WATCHLIST)
    use_scan = st.toggle("ให้ Scanner คัดหุ้นจากตลาดเพิ่ม", value=True)
    top_n = st.slider("จำนวนหุ้นเด่นจาก Scanner", 3, 15, 5, disabled=not use_scan)
    min_value = st.number_input("มูลค่าซื้อขายขั้นต่ำ (ล้านบาท/วัน)", 0.0, 5000.0, 50.0, 10.0)
    with st.expander("รายชื่อหุ้นที่ให้ Scanner สแกน"):
        scan_text = st.text_area("Scan list", ", ".join(DEFAULT_SCAN_LIST), height=150)
    st.divider()
    capital = st.number_input("เงินทุน (บาท)", 0.0, 1e9, 100_000.0, 10_000.0)
    risk_pct = st.slider("ความเสี่ยงต่อไม้ (% ของทุน)", 0.25, 5.0, 1.0, 0.25)
    st.session_state["chart_days"] = st.select_slider("ช่วงกราฟ (วันทำการ)", [60, 120, 180, 250, 500], 180)
    run = st.button("🚀 เริ่มวิเคราะห์", type="primary", use_container_width=True)


# ======================================================================
# Main
# ======================================================================
st.title(f"📈 {APP_TITLE}")
st.caption(f"v{APP_VERSION} · SET/mai · ตัวเลขทุกตัวคำนวณด้วยโค้ดจากข้อมูลจริง · AI ใช้อธิบายและโต้แย้งเท่านั้น")

if run:
    watch = parse_symbols(watch_text)
    scan = [x for x in parse_symbols(scan_text) if x not in watch] if use_scan else []
    universe = watch + scan
    results, failed = {}, []
    bar = st.progress(0.0, text="กำลังดึงข้อมูล...")
    for i, sym in enumerate(universe, 1):
        bar.progress(i / len(universe), text=f"วิเคราะห์ {sym} ({i}/{len(universe)})")
        try:
            r = analyze_symbol(sym, capital, risk_pct)
        except Exception as e:  # หุ้นตัวเดียวพังไม่ให้ทั้งแอปพัง
            r = None
            failed.append(f"{sym}: {type(e).__name__}")
        if r is None:
            if not any(x.startswith(sym + ":") for x in failed):
                failed.append(f"{sym}: ไม่มีข้อมูลราคา")
            continue
        r["in_watchlist"] = sym in watch
        results[sym] = r
    bar.empty()
    picks = sorted(
        [r for r in results.values() if not r["in_watchlist"]
         and r["plan"]["status"] not in ("ระวัง/หลีกเลี่ยง", "ข้อมูลไม่พอ")
         and r["snap"]["avg_value_20d"] >= min_value * 1e6 and r["scores"]["combined"] is not None],
        key=lambda r: r["scores"]["combined"], reverse=True)[:top_n]
    st.session_state.update(results=results, failed=failed, picks=[r["symbol"] for r in picks],
                            watch=[s for s in watch if s in results], scan_ai=None)

results: dict = st.session_state.get("results", {})
if not results:
    st.info("👈 ตั้งค่าแล้วกด **เริ่มวิเคราะห์**")
    st.caption(DISCLAIMER)
    st.stop()

if st.session_state.get("failed"):
    st.warning("ดึงข้อมูลไม่ได้: " + ", ".join(st.session_state["failed"]))

tab_scan, tab_detail, tab_cal = st.tabs(["🏆 หุ้นเด่น & Watchlist", "📊 วิเคราะห์รายตัว", "🗓️ ปฏิทินปันผล & งบ"])

with tab_scan:
    picks, watch = st.session_state.get("picks", []), st.session_state.get("watch", [])
    if picks:
        st.markdown("#### 🏆 หุ้นเด่นจาก Scanner (เรียงตามคะแนนรวม)")
        st.dataframe(pd.DataFrame([summary_row(results[s]) for s in picks]), hide_index=True, use_container_width=True)
        if st.button("🤖 ให้ AI สรุปว่าทำไมหุ้นเหล่านี้น่าสนใจ", disabled=not api_key):
            rows = [{**summary_row(results[s]), "เหตุผลระบบ": results[s]["plan"]["reasons"],
                     "จุดเด่นเทคนิค": results[s]["tech_why"][:4], "จุดเด่นพื้นฐาน": results[s]["fund_why"][:5]}
                    for s in picks]
            with st.spinner("AI กำลังสรุป..."):
                st.session_state["scan_ai"] = ai.scan_brief(api_key, rows)
        sa = st.session_state.get("scan_ai")
        if sa:
            if sa.get("ok"):
                st.success(sa.get("market_view", ""))
                for pk in sa.get("picks", []):
                    st.markdown(f"**{pk.get('symbol')}** — {pk.get('why')}  \n⚠️ {pk.get('watch_out')}")
            else:
                st.error(f"AI ใช้งานไม่ได้: {sa.get('error')}")
    elif st.session_state.get("results"):
        st.caption("Scanner ไม่พบหุ้นที่ผ่านเกณฑ์ (หรือปิด Scanner ไว้)")
    if watch:
        st.markdown("#### ⭐ Watchlist ของคุณ")
        st.dataframe(pd.DataFrame([summary_row(results[s]) for s in watch]), hide_index=True, use_container_width=True)
    st.caption("คะแนนรวม = เทคนิค 45% + พื้นฐาน 40% + R:R 15% · ราคาปัดตามช่วงราคา SET แล้ว")

with tab_detail:
    order = st.session_state.get("picks", []) + st.session_state.get("watch", [])
    order += [s for s in results if s not in order]
    sel = st.selectbox("เลือกหุ้น", order, format_func=lambda s: f"{s} · {results[s]['plan']['status']}")
    if sel:
        render_detail(results[sel], api_key)

with tab_cal:
    st.markdown("#### เหตุการณ์ 180 วันข้างหน้า (Watchlist + หุ้นเด่น)")
    rows = []
    for sym in st.session_state.get("watch", []) + st.session_state.get("picks", []):
        r = results[sym]
        for e in build_timeline(r["fund"], r["div"], r["cal"], []):
            rows.append({"วันที่": e["date"], "อีก (วัน)": e["days_to"], "หุ้น": sym,
                         "เหตุการณ์": e["title"], "สถานะ": f"{e['icon']} {e['status']}", "หมายเหตุ": e["detail"]})
    if rows:
        cal_df = pd.DataFrame(rows).sort_values("วันที่")
        only_div = st.toggle("แสดงเฉพาะปันผล", value=False)
        if only_div:
            cal_df = cal_df[cal_df["เหตุการณ์"].str.contains("ปันผล")]
        st.dataframe(cal_df, hide_index=True, use_container_width=True)
    else:
        st.caption("ไม่มีเหตุการณ์ในช่วงนี้")

st.divider()
st.caption(DISCLAIMER)
