import os
import time
from typing import Dict, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()  # <-- loads .env

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ib_insync import IB, Stock, MarketOrder

from pyngrok import ngrok


# =========================
# CONFIG (ENV VARS)
# =========================
TV_SECRET = os.getenv("TV_SECRET", "CHANGE_ME_TO_A_LONG_RANDOM_SECRET")

# IBKR (TWS or IB Gateway on same machine)
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7496"))         # TWS paper commonly 7497, live 7496
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "7"))

# Trading params
SYMBOL_LOCK = os.getenv("SYMBOL_LOCK", "SPY").upper()
QTY = int(os.getenv("QTY", "10"))
TP1_FRACTION = float(os.getenv("TP1_FRACTION", "0.5"))

# Safety rails
MAX_ABS_POS = int(os.getenv("MAX_ABS_POS", "200"))
COOLDOWN_SEC = float(os.getenv("COOLDOWN_SEC", "1.0"))
DEDUP_TTL_SEC = int(os.getenv("DEDUP_TTL_SEC", "120"))

# Kill switch
ENABLE_TRADING = os.getenv("ENABLE_TRADING", "NO").upper() == "YES"

# Ngrok
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")  # from .env
NGROK_REGION = os.getenv("NGROK_REGION")        # optional


# =========================
# APP + IB
# =========================
app = FastAPI()
ib = IB()

_seen: Dict[Tuple[str, str, str, str, str], float] = {}
_last_order_ts = 0.0


class TVAlert(BaseModel):
    secret: str
    event: str                   # ORB60_BUY, ORB60_SELL, TP1_HIT, TP2_HIT, SL_HIT
    symbol: str                  # SPY
    time: str                    # {{time}}
    tf: str                      # {{interval}}
    price: Optional[float] = None
    alert_name: Optional[str] = ""


def ensure_ib_connected() -> None:
    if ib.isConnected():
        return
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)


def cleanup_dedupe() -> None:
    now = time.time()
    dead = [k for k, ts in _seen.items() if now - ts > DEDUP_TTL_SEC]
    for k in dead:
        _seen.pop(k, None)


def is_duplicate(a: TVAlert) -> bool:
    cleanup_dedupe()
    key = (a.symbol.upper(), a.event.upper(), a.time, a.tf, (a.alert_name or ""))
    if key in _seen:
        return True
    _seen[key] = time.time()
    return False


def enforce_cooldown() -> None:
    global _last_order_ts
    now = time.time()
    if now - _last_order_ts < COOLDOWN_SEC:
        raise HTTPException(status_code=429, detail="cooldown")
    _last_order_ts = now


def current_position(symbol: str) -> int:
    for p in ib.positions():
        c = p.contract
        if getattr(c, "secType", "") == "STK" and getattr(c, "symbol", "").upper() == symbol.upper():
            return int(p.position)
    return 0


def enforce_max_abs_pos(proposed_pos: int) -> None:
    if abs(proposed_pos) > MAX_ABS_POS:
        raise HTTPException(
            status_code=400,
            detail=f"max position exceeded: proposed {proposed_pos}, cap {MAX_ABS_POS}"
        )


def place_market(symbol: str, side: str, qty: int):
    qty = int(qty)
    if qty <= 0:
        raise HTTPException(status_code=400, detail="qty must be > 0")

    contract = Stock(symbol, "ARCA", "USD")
    ib.qualifyContracts(contract)

    order = MarketOrder(side.upper(), qty)
    trade = ib.placeOrder(contract, order)
    ib.sleep(0.5)
    return trade


def flatten(symbol: str, pos: int):
    if pos == 0:
        return None
    if pos > 0:
        return place_market(symbol, "SELL", pos)
    else:
        return place_market(symbol, "BUY", abs(pos))


@app.post("/tv")
def tv_webhook(a: TVAlert):
    if a.secret != TV_SECRET:
        raise HTTPException(status_code=401, detail="bad secret")

    symbol = a.symbol.upper().strip()
    event = a.event.upper().strip()

    if symbol != SYMBOL_LOCK:
        return {"ok": True, "ignored": True, "reason": f"symbol not allowed: {symbol}"}

    allowed = {"ORB60_BUY", "ORB60_SELL", "TP1_HIT", "TP2_HIT", "SL_HIT"}
    if event not in allowed:
        raise HTTPException(status_code=400, detail=f"unknown event: {event}")

    if is_duplicate(a):
        return {"ok": True, "duplicate": True}

    if not ENABLE_TRADING:
        return {"ok": True, "ignored": True, "reason": "ENABLE_TRADING != YES"}

    ensure_ib_connected()
    pos = current_position(symbol)

    enforce_cooldown()

    # SL: flatten immediately (long or short)
    if event == "SL_HIT":
        trade = flatten(symbol, pos)
        return {"ok": True, "event": event, "action": "FLATTEN", "pos_was": pos,
                "orderId": (trade.order.orderId if trade else None)}

    # TP2: flatten remaining long
    if event == "TP2_HIT":
        if pos > 0:
            trade = place_market(symbol, "SELL", pos)
            return {"ok": True, "event": event, "action": "SELL_FLATTEN_LONG", "qty": pos, "pos_was": pos, "pos_now": 0,
                    "orderId": trade.order.orderId}
        return {"ok": True, "event": event, "ignored": True, "reason": f"not long (pos={pos})"}

    # TP1: sell 50% of current long
    if event == "TP1_HIT":
        if pos > 0:
            qty = int(pos * TP1_FRACTION)
            qty = max(1, qty)
            qty = min(qty, pos)
            new_pos = pos - qty
            enforce_max_abs_pos(new_pos)
            trade = place_market(symbol, "SELL", qty)
            return {"ok": True, "event": event, "action": "SELL_PARTIAL", "qty": qty, "pos_was": pos, "pos_now": new_pos,
                    "orderId": trade.order.orderId}
        return {"ok": True, "event": event, "ignored": True, "reason": f"not long (pos={pos})"}

    # ORB60 BUY: add if flat/long; cover-all if short; no flip
    if event == "ORB60_BUY":
        if pos < 0:
            qty = abs(pos)
            trade = place_market(symbol, "BUY", qty) if qty > 0 else None
            return {"ok": True, "event": event, "action": "COVER_ALL", "qty": qty, "pos_was": pos, "pos_now": 0,
                    "orderId": (trade.order.orderId if trade else None)}
        else:
            new_pos = pos + QTY
            enforce_max_abs_pos(new_pos)
            trade = place_market(symbol, "BUY", QTY)
            return {"ok": True, "event": event, "action": "BUY_ADD", "qty": QTY, "pos_was": pos, "pos_now": new_pos,
                    "orderId": trade.order.orderId}

    # ORB60 SELL: add short if flat/short; flatten long if long; no flip
    if event == "ORB60_SELL":
        if pos > 0:
            trade = place_market(symbol, "SELL", pos)
            return {"ok": True, "event": event, "action": "SELL_ALL_LONG", "qty": pos, "pos_was": pos, "pos_now": 0,
                    "orderId": trade.order.orderId}
        else:
            new_pos = pos - QTY
            enforce_max_abs_pos(new_pos)
            trade = place_market(symbol, "SELL", QTY)
            return {"ok": True, "event": event, "action": "SELL_ADD_SHORT", "qty": QTY, "pos_was": pos, "pos_now": new_pos,
                    "orderId": trade.order.orderId}

    return {"ok": True, "event": event}


if __name__ == "__main__":
    # ngrok setup
    if not NGROK_AUTHTOKEN:
        raise RuntimeError("NGROK_AUTHTOKEN is not set. Put it in .env or export it.")

    ngrok.set_auth_token(NGROK_AUTHTOKEN)

    # optional region
    if NGROK_REGION:
        public_url = ngrok.connect(8000, region=NGROK_REGION).public_url
    else:
        public_url = ngrok.connect(8000).public_url

    print("\n✅ TradingView Webhook URL (paste into TradingView):")
    print(f"{public_url}/tv\n")

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)