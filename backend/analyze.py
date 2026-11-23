import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import Groq

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
OUT = ROOT.parent / "docs" / "latest.json"
TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- زنجیره مدل‌های گوگل (به ترتیب اولویت) ---
# طبق گزارش‌ها، مدل‌های 3.7 و 3.8 در سطح رایگان ناپایدار هستند.
# مدل‌های 3.6 و 3.5-lite پایدارتر هستند.
GEMINI_MODEL_CHAIN = [
    "gemini-3.6-flash",       # پایدارترین مدل با کیفیت خوب
    "gemini-3.5-flash-lite",  # سریع‌تر و سبک‌تر
]

def fetch_market_data(symbol):
    """دریافت داده‌های بازار از Twelve Data"""
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval=1day&outputsize=5&apikey={TWELVE_DATA_KEY}"
    )
    try:
        response = requests.get(url, timeout=15)
        data = response.json()
        if "values" not in data or not data["values"]:
            print(f"No data for {symbol}. API response: {data}")
            return None
        values = data["values"][::-1]
        closes = [float(v["close"]) for v in values]
        return {
            "last_close": round(closes[-1], 2),
            "change_pct_5d": round(((closes[-1] / closes[0]) - 1) * 100, 2),
            "closes": [round(c, 2) for c in closes],
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def build_prompt(payload):
    """ساخت پرامپت تحلیل برای هوش مصنوعی"""
    rows = "\n".join(
        f"- {v['label']} ({k}): آخرین بسته {v['data']['last_close']}، "
        f"تغییر ۵روزه {v['data']['change_pct_5d']}٪"
        for k, v in payload.items()
    )
    return (
        "تو یک تحلیل‌گر کوتاه و محافظه‌کار بازار هستی. "
        "برای هر شاخص در ۲ تا ۳ خط، روند و ریسک را به فارسی ساده بگو. "
        "در پایان یک جمع‌بندی کلی ۳ خطی بده. "
        "از پیش‌بینی قطعی قیمت خودداری کن.\n\n"
        f"داده‌ها:\n{rows}"
    )

# --- لایه ۱ و ۲: تلاش با مدل‌های گوگل (با tenacity) ---
@retry(
    stop=stop_after_attempt(3), # حداکثر ۳ بار برای هر مدل
    wait=wait_exponential(multiplier=2, min=2, max=15), # فاصله: 2s, 4s, 8s
    retry=retry_if_exception_type(Exception), # تلاش مجدد برای هر خطایی
    reraise=True
)
def try_gemini_model(client, model_name, prompt):
    """تلاش برای تولید محتوا با یک مدل خاص گوگل"""
    print(f"Attempting with Gemini model: {model_name}")
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text

def generate_gemini_analysis(prompt):
    """تلاش با زنجیره مدل‌های گوگل"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    last_exception = None

    for model_name in GEMINI_MODEL_CHAIN:
        try:
            return try_gemini_model(client, model_name, prompt)
        except Exception as e:
            print(f"Model {model_name} failed after retries: {e}")
            last_exception = e
            continue # برو سراغ مدل بعدی در زنجیره

    # اگر هیچ مدلی جواب نداد، خطای آخر را پرتاب کن
    raise last_exception

# --- لایه ۳: پشتیبان نهایی (Groq) ---
def generate_groq_analysis(prompt):
    """تولید تحلیل با استفاده از Groq به عنوان آخرین راه‌حل"""
    print("Falling back to Groq API...")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", # مدل سریع و رایگان Groq
            temperature=0.2,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Groq fallback also failed: {e}")
        raise e

def main():
    # ۱. دریافت داده‌های بازار
    payload = {}
    for item in CONFIG["indices"]:
        d = fetch_market_data(item["symbol"])
        if d:
            payload[item["symbol"]] = {"label": item["label"], "data": d}

    if not payload:
        raise SystemExit("No market data fetched from Twelve Data.")

    prompt = build_prompt(payload)

    # ۲. تولید تحلیل با زنجیره کامل (Gemini -> Groq)
    analysis_text = ""
    try:
        analysis_text = generate_gemini_analysis(prompt)
        print("Analysis generated successfully with Gemini.")
    except Exception as gemini_error:
        print(f"All Gemini models failed. Last error: {gemini_error}")
        try:
            analysis_text = generate_groq_analysis(prompt)
            print("Analysis generated successfully with Groq fallback.")
        except Exception as groq_error:
            analysis_text = (
                "متأسفانه تحلیل هوش مصنوعی در این لحظه امکان‌پذیر نیست. "
                f"خطای Gemini: {gemini_error}. خطای Groq: {groq_error}"
            )
            print(analysis_text)

    # ۳. ذخیره خروجی
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "analysis": analysis_text,
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
