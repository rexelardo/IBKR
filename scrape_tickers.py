import pandas as pd
import requests
import time
import re
from datetime import date

today = date.today().isoformat()


rows = []
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/121.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://stockanalysis.com/",
})
for i in range(12):
    url = f"https://stockanalysis.com/stocks/?page={i+1}"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    response = r.text
    text=response

    obj_pattern = re.compile(r'\{[^}]*\}')

    
    for obj in obj_pattern.findall(text):
        s = re.search(r's:"([^"]+)"', obj)
        n = re.search(r'n:"([^"]+)"', obj)
        i = re.search(r'industry:"([^"]+)"', obj)
        mc = re.search(r'marketCap:(\d+)', obj)

        if all([s, n, i, mc]):
            rows.append({
                "ticker": s.group(1),
                "name": n.group(1),
                "industry": i.group(1),
                "marketCap": int(mc.group(1)),
            })
    print(rows[-10:])

df = pd.DataFrame(rows)

df.to_csv(f"capital_flows/tickers_{today}.csv", index=False)