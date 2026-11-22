import pandas as pd
from io import StringIO
import requests

def fetch(symbol, period, interval):
    # تبدیل نمادهای یاهو به فرمت Stooq
    stooq_symbol = symbol.replace('^', '').lower()
    if stooq_symbol == 'gspc':
        stooq_symbol = '^spx'
    
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200 or not response.text.startswith('Date'):
            return None
        df = pd.read_csv(StringIO(response.text))
        df = df.tail(5)
        if df.empty:
            return None
        return {
            "last_close": round(float(df["Close"].iloc[-1]), 2),
            "change_pct_5d": round(float(
                (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
            ), 2),
            "closes": [round(float(x), 2) for x in df["Close"].tolist()],
        }
    except Exception:
        return None
