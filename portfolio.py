#!/usr/bin/python3


data['Daily_return_pct'] = data['Close'].pct_change(1)
# same as above
data['Daily_return']=data['Close']/data['Close'].shift(1)-1

# Cumulative Daily return
data['Cumulative_return'] = np.cumsum(data['Daily_return'])

# Cumulative Return (Compound factor)
# The compound return is the rate of
# return, usually expressed as a percentage, that represents the cumulative
# effect that a series of gains or losses has on an original amount of capital
# over a period of time
data['Cumulative_return_Comp'] = (1 + data['Daily_return']).cumprod()



port_standard_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

total_return = (data[-1]-data[0]/data[0])
months = len(data)

annualized_return = ((1+total_return)**(12/months))-1

annualized_return_over_3_years = ((1+total_return)**(1/3))-1

# T is the number of data points per year
sigma_year = sigma_measured * np.sqrt(T)

annualized_vol = data.returns.std() * np.sqrt(252)

risk_free_rate = 0.01
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol

df['perc_ret'] = (1 + df.Daily_rets).cumprod() - 1  # Or df.Daily_rets.add(1).cumprod().sub(1)


daily_pct = df.close.pct_change()
daily_cuml_ret = (1 + daily_pct).cumprod()

plt.scatter(daily_pct.mean(),daily_pct.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')

for label,x,y in zip(daily_pct.columns,
                     daily_pct.mean(),
                     daily_pct.std()):
    plt.annonate(label,
                 xy = (x,y),
                 xytest = (30,-30),
                 textcoords = 'offset points',
                 ha = 'right',
                 va = 'bottom',
                 bbox = dict(boxstyle = 'round,pad=0.5',
                             fc = 'yellow',
                             alpha = 0.5),
                 arrowprops = dict(arrowstyle = '->',
                                   connectionstyle = 'arc3,rad=0'))
    plt.xlim(-0.001,0.003)
    plt.ylim(0.005,0.0275)
    plt.gcf().set_size_inches(8,8)
