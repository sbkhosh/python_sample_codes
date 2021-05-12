#!/usr/bin/python3

import matplotlib.pyplot as plt
import os
import pandas as pd
import re

def read_data(filename):
    if not os.path.isfile(filename):
        print("{} does not exist ".format(filename))
        return
    with open(filename) as filehandle:
        lines = filehandle.readlines()
        lines = filter(lambda x: x.strip(), lines)
    return(lines)

def txt2_df(filename):
    periods = []
    start_dates = []
    end_dates = []
    total_closed_trades = []
    total_net_profit = []
    starting_finishing_balance = []
    total_open_trades = []
    open_pl = []
    total_paid_fees = []
    max_drawdown = []
    annual_return = []
    expectancy = []
    avg_win = []
    avg_loss = []
    ratio_avg_win_avg_loss = []
    percent_profitable = []
    longs = []
    shorts = []
    avg_holding_time = []
    winning_trades_avg_holding_time = []
    losing_trades_avg_holding_time = []
    sharpe_ratio = []
    winning_streak = []
    losing_streak = []
    largest_winning_trade = []
    largest_losing_trade = []
    total_winning_trades = []
    total_losing_trades = []
    market_change = []

    feat = [ "Total Closed Trades", "Total Net Profit", "Starting => Finishing Balance",
             "Total Open Trades", "Open PL", "Total Paid Fees", "Max Drawdown",                    
             "Annual Return", "Expectancy", "Ratio Avg Win / Avg Loss",
             "Percent Profitable", "Avg Holding Time",                
             "Winning Trades Avg Holding Time", "Losing Trades Avg Holding Time",
             "Sharpe Ratio", "Winning Streak", "Losing Streak", "Largest Winning Trade",           
             "Largest Losing Trade", "Total Winning Trades", "Total Losing Trades",             
             "Market Change"]

    var_feat = [ total_closed_trades, total_net_profit,starting_finishing_balance,
                 total_open_trades, open_pl, total_paid_fees, max_drawdown,
                 annual_return, expectancy, ratio_avg_win_avg_loss,
                 percent_profitable, avg_holding_time, winning_trades_avg_holding_time,
                 losing_trades_avg_holding_time, sharpe_ratio, winning_streak, losing_streak,
                 largest_winning_trade, largest_losing_trade, total_winning_trades,
                 total_losing_trades, market_change ]
    
    dct_var_feat = dict(zip(feat,var_feat))
    
    with open(filename) as f:
        lines = f.readlines() # list containing lines of file              
        for line in lines:
            line = line.strip() # remove leading/trailing white spaces
            if line:
                if line.startswith('period'):
                    periods.append(line.split('(')[0].split('|')[1].split('days')[0].strip())
                    
                if line.startswith('starting-ending'):
                    start_dates.append(line.split('|')[1].strip().split('=>')[0].strip())
                    end_dates.append(line.split('|')[1].strip().split('=>')[1])

                if line.startswith('Avg Win | Avg Loss'):
                    avg_win.append(line.split('|')[2].strip())
                    avg_loss.append(line.split('|')[3].strip())
                    
                if line.startswith('Longs | Shorts'):
                    longs.append(line.split('|')[2].strip())
                    shorts.append(line.split('|')[3].strip())

                for k,v in dct_var_feat.items():
                    if line.startswith(k):
                        v.append(line.split('|')[1].strip())

    additions = [{'Period': periods},{'Start Date': start_dates},{'End Date': end_dates},
                 {'Avg Win': avg_win, 'Avg Loss': avg_loss, 'Longs': longs, 'Shorts': shorts}]
    [dct_var_feat.update(el) for el in additions]

    df = pd.DataFrame(dct_var_feat)
    df.set_index('End Date',inplace=True)

    tnp0 = df['Total Net Profit'].apply(lambda x: x.split(' ')[0].replace(',','')).astype(float)
    tnp1 = df['Total Net Profit'].apply(lambda x: x.split(' ')[1].replace('(','').replace(')','').replace('%','')).astype(float)
    df['Total Net Profit'] = tnp0
    df['Total Net Profit (%)'] = tnp1

    sfb0 = df['Starting => Finishing Balance'].apply(lambda x: x.split('=>')[0].replace(',','')).astype(float)
    sfb1 = df['Starting => Finishing Balance'].apply(lambda x: x.split('=>')[1].replace(',','')).astype(float)
    df['Starting Balance'] = sfb0
    df['Ending Balance'] = sfb1

    df['Total Open Trades'] = df['Total Open Trades'].astype(int)
    df['Open PL'] = df['Open PL'].apply(lambda x: x.replace(',','')).astype(float)

    df['Total Paid Fees'] = df['Total Paid Fees'].apply(lambda x: x.replace(',','')).astype(float)
    df['Max Drawdown (%)'] = df['Max Drawdown'].apply(lambda x: x.replace(',','').replace('%','')).astype(float)
    df['Annual Return (%)'] = df['Annual Return'].apply(lambda x: x.replace(',','').replace('%','')).astype(float)

    expt0 = df['Expectancy'].apply(lambda x: x.split(' ')[0].replace(',','')).astype(float)
    expt1 = df['Expectancy'].apply(lambda x: x.split(' ')[1].replace('(','').replace(')','').replace('%','')).astype(float)
    df['Expectancy'] = expt0
    df['Expectancy (%)'] = expt1

    df['Avg Win'] = df['Avg Win'].apply(lambda x: x.replace(',','')).astype(float)
    df['Avg Loss'] = df['Avg Loss'].apply(lambda x: x.replace(',','')).astype(float)

    df['Ratio Avg Win / Avg Loss'] = df['Ratio Avg Win / Avg Loss'].astype(float)
    df['Percent Profitable (%)'] = df['Percent Profitable'].apply(lambda x: x.replace(',','').replace('%','')).astype(float)

    df['Longs (%)'] = df['Longs'].apply(lambda x: x.replace(',','').replace('%','')).astype(float)
    df['Shorts (%)'] = df['Shorts'].apply(lambda x: x.replace(',','').replace('%','')).astype(float)

    df['Avg Holding Time'] = df['Avg Holding Time'].astype(str)
    df['Winning Trades Avg Holding Time'] = df['Winning Trades Avg Holding Time'].astype(str)
    df['Losing Trades Avg Holding Time'] = df['Losing Trades Avg Holding Time'].astype(str)

    df['Sharpe Ratio'] = df['Sharpe Ratio'].apply(lambda x: x.replace(',','')).astype(float)
    df['Winning Streak'] = df['Winning Streak'].astype(int)
    df['Losing Streak'] = df['Losing Streak'].astype(int)

    df['Largest Winning Trade'] = df['Largest Winning Trade'].apply(lambda x: x.replace(',','')).astype(float)
    df['Largest Losing Trade'] = df['Largest Losing Trade'].apply(lambda x: x.replace(',','')).astype(float)
    
    df['Total Winning Trades'] = df['Total Winning Trades'].astype(int)
    df['Total Losing Trades'] = df['Total Losing Trades'].astype(int)

    df['Market Change (%)'] = df['Market Change'].apply(lambda x: x.replace(',','').replace('%','')).astype(float)

    cols_select = ['Total Closed Trades', 'Total Net Profit', 'Total Open Trades', 'Open PL',
                   'Total Paid Fees', 'Ratio Avg Win / Avg Loss', 'Avg Holding Time',
                   'Winning Trades Avg Holding Time', 'Losing Trades Avg Holding Time',
                   'Sharpe Ratio', 'Winning Streak', 'Losing Streak',
                   'Largest Winning Trade', 'Largest Losing Trade', 'Total Winning Trades',
                   'Total Losing Trades', 'Market Change', 'Period', 'Start Date',
                   'Avg Win', 'Avg Loss', 'Total Net Profit (%)',
                   'Starting Balance', 'Ending Balance', 'Max Drawdown (%)',
                   'Annual Return (%)', 'Expectancy (%)', 'Percent Profitable (%)',
                   'Longs (%)', 'Shorts (%)', 'Market Change (%)']

    return(df[cols_select])

if __name__ == '__main__':
    filename = 'nohup.out'
    df = txt2_df(filename)
