import websocket, json, pprint

SOCKET = "wss://stream.binance.us:9443/ws/dogeusd@trade"

def on_open(ws):
	print('opened connection')

def on_close(ws):
	print('closed connection')

def on_message(ws, message):
    print('received message')
    json_message = json.loads(message)
    pprint.pprint(json_message)

ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()
