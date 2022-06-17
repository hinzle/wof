from binance.client import Client
import config

client = Client(config.API_KEY, config.API_SECRET, testnet=True, tld='us')
