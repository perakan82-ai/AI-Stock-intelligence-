"""
ai.py — LLM layer (Gemini) สำหรับงานเชิงคุณภาพเท่านั้น
- วิเคราะห์ Sentiment ข่าว, สรุปจุดเด่น/ความเสี่ยง, Bull vs Bear Debate, เงื่อนไขที่ทำให้ Thesis ผิด
- ห้าม AI สร้างตัวเลขใหม่: ส่งเฉพาะตัวเลขที่คำนวณแล้วเข้าไป และสั่งให้อ้างอิงได้เฉพาะตัวเลขในข้อมูลนั้น
- สถานะ/จุดเข้า-ออก มาจาก signals.py (โค้ด) — AI ทำหน้าที่อธิบายและโต้แย้งเท่านั้น
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from config import AI_TEMPERATURE, GEMINI_MODEL

SYSTEM_RULES = """คุณคือทีมนักวิเคราะห์หุ้นไทย 6 ตำแหน่ง: Fundamental, Technical, Smart Money/Flow, News, Risk และ Chief Investment Analyst
กฎเหล็ก:
1. ใช้เฉพาะตัวเลขที่อยู่ใน DATA เท่านั้น ห้ามสร้างตัวเลข ราคา เป้าหมาย หรือวันที่ใหม่ขึ้นเอง
2. ถ้าข้อมูลส่วนใดเป็น null ให้บอกว่า "ไม่มีข้อมูล" ห้ามเดา
3. จุดเข้า/จุดตัดขาดทุน/เป้าขาย คำนวณโดยระบบแล้ว ให้อธิบายเหตุผลและความเสี่ยง ไม่ต้องคิดใหม่
4. แยก "ข้อเท็จจริง" กับ "ความเห็น" ให้ชัด เหตุการณ์สถานะ Estimated คือการประมาณ ไม่ใช่ข้อเท็จจริง
5. ตอบเป็นภาษาไทย กระชับ ตรงประเด็น ใช้ศัพท์การเงินไทยที่ถูกต้อง
6. ตอบเป็น JSON ตาม schema ที่กำหนดเท่านั้น"""

STOCK_SCHEMA = """{
  "news_sentiment": {"overall": "บวก|ลบ|กลาง|ผสม", "items": [{"title": "...", "sentiment": "บวก|ลบ|กลาง", "impact": "ผลต่อรายได้/กำไร/ราคา สั้นๆ"}]},
  "fundamental_view": "2-3 ประโยค",
  "technical_view": "2-3 ประโยค อ้างอิงแนวรับ/ต้าน EMA RSI จาก DATA",
  "bull_case": ["เหตุผลฝั่งกระทิง 3-4 ข้อ"],
  "bear_case": ["เหตุผลฝั่งหมี 3-4 ข้อ"],
  "catalysts": ["ปัจจัยหนุนที่ต้องติดตาม รวมเหตุการณ์ใน timeline"],
  "thesis_invalidation": ["เงื่อนไขที่ถ้าเกิดขึ้นแปลว่าเหตุผลการซื้อไม่เป็นจริงแล้ว"],
  "event_strategy": "ควรเตรียมตัวอย่างไรก่อนเหตุการณ์ที่ใกล้ที่สุด (งบ/XD/AGM)",
  "chief_verdict": {"agree_with_system_status": true, "comment": "ความเห็น Chief Analyst หลังชั่งน้ำหนัก Bull vs Bear 2-4 ประโยค"},
  "summary": "สรุป 1 ย่อหน้าสำหรับนักลงทุน"
}"""


def _json_default(o: Any):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def build_payload(sym: str, fund: dict, snap: dict, plan: dict, scores: dict,
                  div_out: dict, timeline: list[dict], news: list[dict]) -> dict:
    """ย่อข้อมูลที่คำนวณแล้วให้เหลือเฉพาะที่ AI ต้องใช้"""
    f_keys = ["name", "sector", "industry", "pe", "forward_pe", "pbv", "roe", "net_margin", "debt_to_equity",
              "current_ratio", "revenue_yoy", "earnings_yoy", "fcf_ttm", "payout_ratio", "market_cap",
              "target_mean_price", "analyst_count"]
    s_keys = ["price", "date", "change_pct", "ema20", "ema50", "ema200", "rsi", "macd_hist", "atr_pct",
              "vol_ratio", "ret_1m", "ret_3m", "ret_6m", "structure", "patterns", "breakout_20d",
              "high_52w", "low_52w", "dist_52w_high_pct", "avg_value_20d"]
    p = plan["primary"]
    return _round({
        "symbol": sym,
        "fundamental": {k: fund.get(k) for k in f_keys},
        "business_summary": (fund.get("summary") or "")[:800] or None,
        "technical": {**{k: snap.get(k) for k in s_keys},
                      "supports": [z["price"] for z in snap["supports"]],
                      "resistances": [z["price"] for z in snap["resistances"]]},
        "system_trade_plan": {
            "status": plan["status"], "reasons": plan["reasons"], "setup": p["name"],
            "entry_zone": [p["entry_low"], p["entry_high"]], "stop": p["stop"],
            "targets": [{"price": t["price"], "rr": t["rr"], "basis": t["basis"]} for t in p["targets"]],
        },
        "scores": scores,
        "dividend": {k: div_out.get(k) for k in ("ttm_dps", "ttm_yield", "avg_yield_3y", "per_year", "streak_years")}
                    | {"next": [{"xd": n["xd"], "status": n["status"], "last_amount": n.get("amount")}
                                for n in div_out.get("next", [])]},
        "upcoming_events": [{"date": e["date"], "days_to": e["days_to"], "title": e["title"], "status": e["status"]}
                            for e in timeline if e["status"] != "News"][:8],
        "news_headlines": [{"title": n["title"], "date": n.get("published"), "source": n.get("source")}
                           for n in news[:10]],
    })


def _client(api_key: str):
    from google import genai  # import ช้าเพื่อให้แอปเปิดได้แม้ยังไม่ใส่ key
    return genai.Client(api_key=api_key)


def _generate_json(api_key: str, prompt: str) -> dict:
    from google.genai import types
    resp = _client(api_key).models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_RULES,
            temperature=AI_TEMPERATURE,
            response_mime_type="application/json",
        ),
    )
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


def analyze_stock(api_key: str, payload: dict) -> dict:
    prompt = (
        "วิเคราะห์หุ้นตาม DATA ด้านล่าง ให้แต่ละ Agent ให้มุมมอง แล้วจัด Bull vs Bear Debate "
        "ก่อนให้ Chief Investment Analyst สรุป\n\n"
        f"SCHEMA:\n{STOCK_SCHEMA}\n\nDATA:\n{json.dumps(payload, ensure_ascii=False, default=_json_default)}"
    )
    try:
        return {"ok": True, **_generate_json(api_key, prompt)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def scan_brief(api_key: str, rows: list[dict]) -> dict:
    """สรุปภาพรวมผลสแกน: ทำไมหุ้นอันดับต้นน่าสนใจ (อิงตัวเลขจากระบบ)"""
    schema = '{"market_view": "...", "picks": [{"symbol": "...", "why": "...", "watch_out": "..."}]}'
    prompt = (
        "จาก DATA ผลสแกนหุ้นที่ระบบจัดอันดับแล้ว อธิบายเหตุผลที่หุ้นแต่ละตัวน่าสนใจและสิ่งที่ต้องระวัง "
        "ห้ามเปลี่ยนลำดับหรือสร้างตัวเลขใหม่\n\n"
        f"SCHEMA:\n{schema}\n\nDATA:\n{json.dumps(_round(rows), ensure_ascii=False, default=_json_default)}"
    )
    try:
        return {"ok": True, **_generate_json(api_key, prompt)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
