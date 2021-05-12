#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import talib

def compute_rsi():
    raw_data = obj_reader.external_data
    data = raw_data['_'.join(['eth'+'usdt','4h'])]
    rsi = talib.RSI(data["close"])

    fig = plt.figure(figsize=(32,20))
    fig.subplots_adjust(hspace=0.2)

    ax1 = fig.add_subplot(211)
    ax1.plot(data.index, [70] * len(data.index), label="overbought")
    ax1.plot(data.index, [30] * len(data.index), label="oversold")
    ax1.plot(data.index, rsi, label="rsi")
    ax1.legend()
    
    ax2 = fig.add_subplot(212,sharex=ax1)
    ax2.plot(data.index, data['close'], label="close")
    ax2.legend()
    plt.show()
    
