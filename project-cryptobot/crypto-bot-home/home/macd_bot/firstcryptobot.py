# The Crypto Creaker 1.0.b
# Forest Hensley
# All rights reserved.
# 20210707

import config, websocket, json, pprint, talib, numpy

from binance.client import Client
from binance.enums import *


TRADE_SYMBOL = 'ETHUSD'

TRADE_QUANTITY = 0.0

SOCKET = "wss://stream.binance.com:9443/ws/dogeusdt@trade"

print('TRADE_QUANTITY')
# Data Stream stuff- Binance data stream
# these are functions that are sent to the 
# websocket and ouput the message corresponding 
# to the acitivity on the server side.


def on_open(ws):
    print('opened connection')

def on_close(ws):
    print('closed connection')

# Store Data - 

def on_message(ws, message):

    global closes, in_position
    
    print('received message')
    json_message = json.loads(message)
    pprint.pprint(json_message)

# Analyze Data - 

#price=


# Order - executes the order on Binance

client = Client(config.API_KEY, config.API_SECRET, testnet=True, tld='us')

def order(side, quantity, symbol,order_type=ORDER_TYPE_MARKET):
    try:
        print("sending order")
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        print(order)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False

    return True



ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()







