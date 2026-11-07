"""
signals.py — Trade Plan Engine (deterministic)
คำนวณ: โซนเข้าซื้อ, จุดตัดขาดทุน, เป้าขาย 3 ระดับ, R:R, ขนาดไม้ (board lot 100), สถานะหุ้น, คะแนนรวม
ทุกราคาถูกปัดตามช่วงราคา (tick size) ของ SET
"""
from __future__ import annotations

import math

from config import ATR_STOP_MULT, BOARD_LOT, MIN_RR_FOR_BUY, round_to_tick, tick_size


def _targets(entry: float, stop: float, resistances: list[dict], above: float) -> list[dict]:
    """เป้าขายจากแนวต้านจริงก่อน ถ้าไม่พอใช้ R-multiple (ระบุชัดว่าเป็นเป้าคำนวณ)"""
    risk = entry - stop
    tps = []
    for r in resistances:
        if r["price"] > above and len(tps) < 3:
            # ขายก่อนชนแนวต้าน 1 tick เพื่อให้ได้คิว
            px = round_to_tick(r["price"] - tick_size(r["price"]), "down")
            if px > entry and (not tps or px > tps[-1]["price"]):
                tps.append({"price": px, "basis": f"แนวต้านเดิม (ทดสอบ {r['touches']} ครั้ง)"})
    mult = 2.0
    while len(tps) < 3 and risk > 0:
        px = round_to_tick(entry + mult * risk, "down")
        if not tps or px > tps[-1]["price"]:
            tps.append({"price": px, "basis": f"เป้าคำนวณ {mult:.0f}R (ไม่มีแนวต้านเดิมเหนือราคา)"})
        mult += 1.0
    for tp in tps:
        tp["rr"] = round((tp["price"] - entry) / risk, 2) if risk > 0 else None
        tp["pct"] = (tp["price"] / entry - 1) * 100
    return tps


def build_trade_plan(s: dict, capital: float = 0.0, risk_pct: float = 1.0) -> dict:
    p, atr = s["price"], s["atr"] or s["price"] * 0.02
    sup, res = s["supports"], s["resistances"]
    ema20, ema50, ema200 = s["ema20"], s["ema50"], s["ema200"]
    uptrend_major = ema200 is not None and p > ema200
    struct_down = s["structure"].startswith("ขาลง")
    downtrend = struct_down and ((ema200 is not None and p < ema200) or (ema50 is not None and p < ema50))
    trend_ok = (ema50 is None or p > ema50) and not struct_down  # เงื่อนไขขั้นต่ำก่อนรอเบรก

    # ---------- แผน A: ซื้อเมื่อย่อ (Pullback to support) ----------
    near_sup = [z for z in sup if p - z["price"] <= 3 * atr]
    if near_sup:
        base, base_note = near_sup[0]["price"], f"แนวรับ Price Action (ทดสอบ {near_sup[0]['touches']} ครั้ง)"
    elif ema20 and ema20 < p:
        base, base_note = ema20, "เส้น EMA20 (แนวรับเคลื่อนที่)"
    elif ema50 and ema50 < p:
        base, base_note = ema50, "เส้น EMA50 (แนวรับเคลื่อนที่)"
    elif sup:
        base, base_note = sup[0]["price"], f"แนวรับถัดไป (ทดสอบ {sup[0]['touches']} ครั้ง)"
    else:
        base, base_note = s["low_52w"], "จุดต่ำสุด 52 สัปดาห์"

    entry_low = round_to_tick(base, "up")
    entry_high = round_to_tick(base + 0.5 * atr, "down")
    entry_high = max(entry_high, entry_low)
    entry_mid = round_to_tick((entry_low + entry_high) / 2)
    stop = round_to_tick(base - ATR_STOP_MULT * atr, "down")
    pull_tps = _targets(entry_mid, stop, res, entry_high)
    pull = {
        "name": "ซื้อเมื่อย่อ (Buy on Pullback)",
        "entry_low": entry_low, "entry_high": entry_high, "entry_ref": entry_mid,
        "stop": stop, "stop_pct": (stop / entry_mid - 1) * 100,
        "basis": base_note, "targets": pull_tps,
        "rr1": pull_tps[0]["rr"] if pull_tps else None,
        "in_zone": entry_low <= p <= entry_high,
    }

    # ---------- แผน B: ซื้อเมื่อเบรก (Breakout) ----------
    brk = None
    if trend_ok and res and (res[0]["price"] - p) <= 2 * atr:
        r1 = res[0]["price"]
        b_entry = round_to_tick(r1 + tick_size(r1), "up")
        b_stop = round_to_tick(r1 - ATR_STOP_MULT * atr, "down")
        b_tps = _targets(b_entry, b_stop, res[1:], b_entry)
        brk = {
            "name": "ซื้อเมื่อยืนเหนือแนวต้าน (Breakout)",
            "entry_low": b_entry, "entry_high": round_to_tick(b_entry + 0.3 * atr, "down"),
            "entry_ref": b_entry, "stop": b_stop, "stop_pct": (b_stop / b_entry - 1) * 100,
            "basis": f"แนวต้าน {r1:.2f} (ทดสอบ {res[0]['touches']} ครั้ง) — รอปิดเหนือแนวพร้อม Volume > 1.5 เท่า",
            "targets": b_tps, "rr1": b_tps[0]["rr"] if b_tps else None, "in_zone": False,
        }
    elif s["breakout_20d"] and (s["vol_ratio"] or 0) >= 1.5 and ema50 and p > ema50:
        # เพิ่งเบรกจุดสูงสุด 20 วันพร้อม volume — เข้าที่ราคาปัจจุบัน
        b_stop = round_to_tick(max(p - 1.5 * atr, entry_low - ATR_STOP_MULT * atr), "down")
        b_entry = round_to_tick(p)
        b_tps = _targets(b_entry, b_stop, res, b_entry)
        brk = {
            "name": "ตามน้ำหลังเบรก High 20 วัน (Momentum)",
            "entry_low": b_entry, "entry_high": round_to_tick(p + 0.3 * atr, "down"),
            "entry_ref": b_entry, "stop": b_stop, "stop_pct": (b_stop / b_entry - 1) * 100,
            "basis": f"ปิดเหนือ High 20 วันพร้อม Volume {s['vol_ratio']:.1f} เท่า",
            "targets": b_tps, "rr1": b_tps[0]["rr"] if b_tps else None, "in_zone": True,
        }

    # ---------- สถานะ ----------
    reasons = []
    if downtrend:
        status = "ระวัง/หลีกเลี่ยง"
        reasons.append("โครงสร้างเป็น Lower High/Lower Low และราคาต่ำกว่าเส้นค่าเฉลี่ยหลัก — ยังไม่ควรรับมีด")
    elif pull["in_zone"] and (pull["rr1"] or 0) >= MIN_RR_FOR_BUY and uptrend_major:
        status = "เข้าซื้อได้ (อยู่ในโซนซื้อ)"
        reasons.append(f"ราคาอยู่ในโซนซื้อ {entry_low:.2f}–{entry_high:.2f} และ R:R ถึงเป้าแรก {pull['rr1']:.1f} เท่า")
    elif brk and brk["in_zone"] and (brk["rr1"] or 0) >= MIN_RR_FOR_BUY:
        status = "เข้าซื้อได้ (อยู่ในโซนซื้อ)"
        reasons.append(brk["basis"])
    elif brk:
        status = "รอเบรกแนวต้าน"
        reasons.append(f"ราคาใกล้แนวต้าน — ซื้อเมื่อปิดเหนือ {brk['entry_low']:.2f}")
    elif p > entry_high and uptrend_major and not struct_down:
        status = "รอย่อเข้าโซนซื้อ"
        reasons.append(f"แนวโน้มดีแต่ราคาห่างโซนซื้อ {(p / entry_high - 1) * 100:.1f}% — ไม่ไล่ราคา")
    elif ema200 is None:
        status = "ข้อมูลไม่พอ"
        reasons.append("ประวัติราคาไม่ถึง 200 วัน คำนวณแนวโน้มหลักไม่ได้")
    else:
        status = "ถือ/ติดตาม"
        reasons.append("ยังไม่มีสัญญาณเข้าที่ได้เปรียบชัดเจน")

    if s["rsi"] is not None and s["rsi"] > 75:
        reasons.append(f"RSI {s['rsi']:.0f} ร้อนแรงมาก เสี่ยงย่อตัวระยะสั้น")
    if s["avg_value_20d"] < 20_000_000:
        reasons.append(f"สภาพคล่องต่ำ (มูลค่าซื้อขายเฉลี่ย {s['avg_value_20d']/1e6:.1f} ล้านบาท/วัน)")

    primary = brk if (brk and brk["in_zone"]) else pull
    sizing = position_size(primary["entry_ref"], primary["stop"], capital, risk_pct)

    exit_rules = [
        f"ตัดขาดทุนทันทีถ้าราคาปิดต่ำกว่า {primary['stop']:.2f} บาท ({primary['stop_pct']:.1f}%)",
        "ขายทำกำไรแบ่งไม้: 1/3 ที่ TP1, 1/3 ที่ TP2, ที่เหลือที่ TP3 หรือ Trailing stop",
        "เมื่อถึง TP1 ให้เลื่อนจุดตัดขาดทุนขึ้นมาที่ราคาทุน (Breakeven)",
    ]
    if ema50:
        exit_rules.append(f"หากราคาปิดต่ำกว่า EMA50 ({ema50:.2f}) 2 วันติด ให้ทบทวน/ลดสถานะ")
    if s["rsi"] is not None:
        exit_rules.append("หาก RSI > 75 แล้วเกิดแท่ง Bearish Engulfing/Shooting star ให้ขายทำกำไรบางส่วน")

    return {"status": status, "reasons": reasons, "pullback": pull, "breakout": brk,
            "primary": primary, "sizing": sizing, "exit_rules": exit_rules}


def position_size(entry: float, stop: float, capital: float, risk_pct: float) -> dict | None:
    """จำนวนหุ้นตามความเสี่ยงที่รับได้ ปัดลงเป็น board lot 100 หุ้น"""
    if not capital or capital <= 0 or entry <= 0 or stop >= entry:
        return None
    risk_amount = capital * risk_pct / 100
    per_share = entry - stop
    by_risk = math.floor(risk_amount / per_share / BOARD_LOT) * BOARD_LOT
    by_cash = math.floor(capital / entry / BOARD_LOT) * BOARD_LOT
    shares = max(min(by_risk, by_cash), 0)
    return {
        "shares": shares, "lots": shares // BOARD_LOT, "cost": shares * entry,
        "max_loss": shares * per_share, "risk_amount": risk_amount,
        "capped_by_cash": by_cash < by_risk,
    }


def combined_score(tech: float | None, fund: float | None, rr1: float | None) -> dict:
    """คะแนนรวม = เทคนิค 45% + พื้นฐาน 40% + R:R 15% (ส่วนที่ไม่มีข้อมูลตัดออกและเฉลี่ยน้ำหนักใหม่)"""
    parts = []
    if tech is not None:
        parts.append((0.45, tech))
    if fund is not None:
        parts.append((0.40, fund))
    if rr1 is not None:
        parts.append((0.15, min(max(rr1, 0) / 3, 1) * 100))
    if not parts:
        return {"score": None, "complete": False}
    w = sum(a for a, _ in parts)
    return {"score": round(sum(a * b for a, b in parts) / w, 1), "complete": fund is not None and tech is not None}
