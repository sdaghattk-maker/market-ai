import json
import os
from datetime import datetime, timezone
from pathlib import Path

import google.generativeai as genai
import yfinance as yf

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
OUT = ROOT.parent / "docs" / "latest.json"

def fetch(symbol, period, interval):
    t = yf.Ticker(symbol)
    hist = t.history(period=period, interval=interval)
    if hist.empty:
        return None
    tail = hist.tail(5)
    return {
        "last_close": round(float(tail["Close"].iloc[-1]), 2),
        "change_pct_5d": round(float(
            (tail["Close"].iloc[-1] / tail["Close"].iloc[0] - 1) * 100
        ), 2),
        "closes": [round(float(x), 2) for x in tail["Close"].tolist()],
    }

def build_prompt(payload, lang):
    rows = "\n".join(
        f"- {v['label']} ({k}): آخرین بسته {v['data']['last_close']}، "
        f"تغییر ۵روزه {v['data']['change_pct_5d']}٪، سری {v['data']['closes']}"
        for k, v in payload.items()
    )
    if lang == "fa":
        return (
            "تو یک تحلیل‌گر کوتاه و محافظه‌کار بازار هستی. "
            "برای هر شاخص در ۲ تا ۳ خط، روند و ریسک را به فارسی ساده بگو. "
            "در پایان یک جمع‌بندی کلی ۳ خطی بده. "
            "از پیش‌بینی قطعی قیمت خودداری کن.\n\n"
            f"داده‌ها:\n{rows}"
        )
    return (
        "You are a concise, conservative market analyst. "
        "For each index, summarize trend and risk in 2-3 lines. "
        "End with a 3-line overall summary. Avoid deterministic price forecasts.\n\n"
        f"Data:\n{rows}"
    )

def main():
    payload = {}
    for item in CONFIG["indices"]:
        d = fetch(item["symbol"], CONFIG["period"], CONFIG["interval"])
        if d:
            payload[item["symbol"]] = {"label": item["label"], "data": d}

    if not payload:
        raise SystemExit("No market data fetched.")

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(build_prompt(payload, CONFIG["language"]))

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
