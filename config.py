"""
config.py — ค่าคงที่และกฎตลาดหุ้นไทย (SET / mai)
ทุกค่าที่นี่เป็น "กฎ" หรือ "ค่าตั้งต้น" ไม่ใช่ข้อมูลราคา — ข้อมูลราคาจริงดึงจาก market.py เท่านั้น
"""
from __future__ import annotations

APP_VERSION = "1.0.0"
APP_TITLE = "Thai Stock AI Intelligence Lab"

# ---------- AI ----------
GEMINI_MODEL = "gemini-2.5-flash"      # เปลี่ยนเป็นรุ่นอื่นได้
AI_TEMPERATURE = 0.3

# ---------- Data ----------
HISTORY_PERIOD = "2y"                  # ต้องพอสำหรับ EMA200
CACHE_TTL_PRICE = 15 * 60              # วินาที
CACHE_TTL_FUNDAMENTAL = 6 * 60 * 60
CACHE_TTL_NEWS = 30 * 60
DATA_SOURCE_NAME = "Yahoo Finance (yfinance) + Google News RSS"

# รายชื่อหุ้นตั้งต้นสำหรับ Scanner (หุ้นใหญ่สภาพคล่องสูง — แก้ไขได้จากหน้าแอป)
# หมายเหตุ: นี่ไม่ใช่รายชื่อ SET50 อย่างเป็นทางการ ซึ่งเปลี่ยนทุกครึ่งปี
DEFAULT_SCAN_LIST = [
    "ADVANC", "AOT", "AWC", "BANPU", "BBL", "BDMS", "BEM", "BGRIM", "BH", "BJC",
    "BTS", "CBG", "COM7", "CPALL", "CPF", "CPN", "CRC", "DELTA", "EA", "EGCO",
    "GPSC", "GULF", "HMPRO", "IVL", "KBANK", "KKP", "KTB", "KTC", "LH", "MINT",
    "MTC", "OR", "OSP", "PTT", "PTTEP", "PTTGC", "RATCH", "SAWAD", "SCB", "SCC",
    "SCGP", "TISCO", "TOP", "TRUE", "TTB", "TU", "WHA",
]
DEFAULT_WATCHLIST = "PTT,CPALL,ADVANC,KBANK,AOT"

# ---------- กฎการซื้อขาย SET ----------
BOARD_LOT = 100  # 1 board lot = 100 หุ้น

# ช่วงราคา (Price Spread / Tick size) ของ SET: (ราคาต่ำกว่า, tick)
SET_TICK_TABLE = [
    (2.0, 0.01),
    (5.0, 0.02),
    (10.0, 0.05),
    (25.0, 0.10),
    (100.0, 0.25),
    (200.0, 0.50),
    (400.0, 1.00),
    (float("inf"), 2.00),
]


def tick_size(price: float) -> float:
    for upper, tick in SET_TICK_TABLE:
        if price < upper:
            return tick
    return 2.00


def round_to_tick(price: float, mode: str = "nearest") -> float:
    """ปัดราคาให้ตรงช่วงราคาที่ซื้อขายได้จริงใน SET (mode: nearest | down | up)"""
    if price is None or price <= 0:
        return price
    t = tick_size(price)
    q = price / t
    if mode == "down":
        n = int(q + 1e-9)
    elif mode == "up":
        n = int(q) if abs(q - int(q)) < 1e-9 else int(q) + 1
    else:
        n = round(q)
    return round(n * t, 2)


def to_yahoo_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    return s if s.endswith(".BK") else f"{s}.BK"


def to_set_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(".BK", "")


# ---------- กำหนดส่งงบการเงิน (ประกาศ ก.ล.ต./ตลท.) ----------
# งบไตรมาส: ภายใน 45 วันหลังสิ้นไตรมาส | งบปี: ภายใน 60 วันหลังสิ้นปีบัญชี (กรณีส่งงบไตรมาส)
QUARTERLY_DEADLINE_DAYS = 45
ANNUAL_DEADLINE_DAYS = 60
# ประชุมสามัญผู้ถือหุ้น: ภายใน 4 เดือนหลังสิ้นปีบัญชี (พ.ร.บ.บริษัทมหาชนจำกัด)
AGM_DEADLINE_MONTHS = 4

# ---------- ค่าตั้งต้นการวางแผนเทรด ----------
ATR_STOP_MULT = 1.0          # stop = แนวรับ - 1 ATR
MIN_RR_FOR_BUY = 1.5         # R:R ขั้นต่ำถึงจะนับว่า "เข้าซื้อได้"
PIVOT_WINDOW = 5
SR_LOOKBACK_DAYS = 250

STATUS_COLORS = {
    "เข้าซื้อได้ (อยู่ในโซนซื้อ)": "🟢",
    "รอเบรกแนวต้าน": "🔵",
    "รอย่อเข้าโซนซื้อ": "🟡",
    "ถือ/ติดตาม": "⚪",
    "ระวัง/หลีกเลี่ยง": "🔴",
    "ข้อมูลไม่พอ": "⚫",
}

DISCLAIMER = (
    "⚠️ แอปนี้เป็นเครื่องมือช่วยวิเคราะห์ (Decision Support Tool) ไม่ใช่คำแนะนำการลงทุนเฉพาะบุคคล "
    "ตัวเลขทั้งหมดคำนวณจากข้อมูล Yahoo Finance ซึ่งอาจล่าช้าหรือคลาดเคลื่อน "
    "วัน XD / วันจ่ายปันผล / วันประกาศงบ ที่เป็น 'ประมาณการ' ต้องยืนยันกับประกาศบน set.or.th ก่อนตัดสินใจทุกครั้ง"
)
