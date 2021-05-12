from jesse import register_custom_exception_handler
from jesse.modes import import_candles_mode
from jesse.services import db
 
###################
#     SETTINGS    #
###################
symbols = [
    "BTC-USDT",
    "ETH-USDT",
    "LTC-USDT"
]
 
exchange = 'Binance Futures'
start_date = '2019-01-01'
timeframe = '15m'
strategy = 'BB_Beta'
 
 
 
 
def inject_local_routes(routes_, extra_candles_):
    """
    injects routes from local routes folder
    """
    from jesse.routes import router
    router.set_routes(routes_)
    router.set_extra_candles(extra_candles_)
 
 
def get_candles(start_date, routes_, extra_candles_):
    inject_local_routes(routes_=routes_, extra_candles_=extra_candles_)
 
    for i in symbols:
        try:
            tmfrm = [x[2] for x in routes_ if x[1] == i][0]
            print('Downloading:', i, tmfrm)
 
            register_custom_exception_handler()
            import_candles_mode.run(exchange=exchange, symbol=i, start_date_str=start_date, skip_confirmation=True)
            db.close_connection()
        except IndexError:
            print(f'Please add {i} to routes_')
        except Exception as e:
            print(e)
 
 
if __name__ == '__main__':
 
    get_candles(start_date=start_date, routes_=[(exchange, x, timeframe, strategy) for x in symbols], extra_candles_=[])
