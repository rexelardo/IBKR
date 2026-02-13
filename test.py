from ib_insync import *
import pandas as pd
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)

print("Connected:", ib.isConnected())
print("Server time:", ib.reqCurrentTime())
print("Accounts:", ib.managedAccounts())



contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 M',
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=1
)

df = util.df(bars)
print(df.tail())