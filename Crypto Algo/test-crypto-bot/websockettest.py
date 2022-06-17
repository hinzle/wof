import config, websocket, json, pprint, talib, numpy

from binance.client import Client

SOCKET = "wss://stream.binance.us:9443/ws/dogeusd@trade"

client = Client(config.API_KEY, config.API_SECRET, testnet=True, tld='us')

def on_open(ws):
    print('opened connection')

def on_close(ws):
    print('closed connection')

def on_message(ws, message):
    global doge

    print('received message')
    json_message = json.loads(message)
    pprint.pprint(json_message)
    
    text=open("test.txt", "w")
    text.writelines(json_message)
    text.close()
    
ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()