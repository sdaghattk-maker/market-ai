API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
OUT = ROOT.parent / "docs" / "latest.json"
API_KEY = os.environ.get("TWELVE_DATA_API_KEY")

def fetch(symbol):
    # نمادهای Twelve Data معمولاً بدون ^ هستند
    clean_symbol = symbol.replace("^", "")
    url = f"https://api.twelvedata.com/time_series?symbol={clean_symbol}&interval=1day&outputsize=5&apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=15)
        data = response.json()
        if "values" not in data or not data["values"]:
            print(f"No data for {symbol}. API response: {data}")
            return None
        # Twelve Data داده‌ها را از جدید به قدیم برمی‌گرداند
        values = data["values"][::-1] # معکوس کردن برای ترتیب زمانی
        closes = [float(v["close"]) for v in values]
        return {
            "last_close": round(closes[-1], 2),
            "change_pct_5d": round(((closes[-1] / closes[0]) - 1) * 100, 2),
            "closes": [round(c, 2) for c in closes],
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def main():
    payload = {}
    for item in CONFIG["indices"]:
        d = fetch(item["symbol"])
        if d:
            payload[item["symbol"]] = {"label": item["label"], "data": d}

    if not payload:
        raise SystemExit("No market data fetched from Twelve Data.")

    # بخش هوش مصنوعی (بدون تغییر نسبت به قبل)
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    rows = "\n".join(
        f"- {v['label']} ({k}): آخرین بسته {v['data']['last_close']}، تغییر ۵روزه {v['data']['change_pct_5d']}٪"
        for k, v in payload.items()
    )
    prompt = f"تو یک تحلیل‌گر کوتاه و محافظه‌کار بازار هستی. برای هر شاخص در ۲ تا ۳ خط، روند و ریسک را به فارسی ساده بگو. در پایان یک جمع‌بندی کلی ۳ خطی بده. از پیش‌بینی قطعی قیمت خودداری کن.\n\nداده‌ها:\n{rows}"
    
    resp = model.generate_content(prompt)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "analysis": resp.text,
                "raw": payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote", OUT)

if __name__ == "__main__":
    main()
