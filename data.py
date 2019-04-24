#!/usr/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.ExcelFile("obs.xls")
# print data.sheet_names

# Read 2nd section, by age
data_age = data.parse(u'7.5', skiprows=4, skipfooter=14)
print data_age

# Rename Unnamed to year
# data_age.rename(columns={u'Unnamed: 0': u'Year'}, inplace=True)

# Drop empties and reset index
# data_age.dropna(inplace=True)
# data_age.set_index('Year', inplace=True)

# print "After Clean up:"
# print data_age

# plot
# data_age.plot()
# plt.show()

# Plotting everything cause total to override everything. So drop it.
# Drop the total column and plot
# data_age_minus_total = data_age.drop('Total', axis=1)
# data_age_minus_total.plot()
# plt.show()
# plt.close()

# Plot children vs adults
# data_age['Under 16'].plot(label="Under 16")
# data_age['35-44'].plot(label="35-44")
# plt.legend(loc="upper left")
# plt.show()

# kids_values = data_age['Under 16'].values

# x_axis = range(len(kids_values))

# poly_degree_1 = 3
# curve_fit_1 = np.polyfit(x_axis, kids_values, poly_degree_1)
# poly_interp_1 = np.poly1d(curve_fit_1)
# poly_fit_values_1 = []

# poly_degree_2 = 4
# curve_fit_2 = np.polyfit(x_axis, kids_values, poly_degree_2)
# poly_interp_2 = np.poly1d(curve_fit_2)
# poly_fit_values_2 = []

# poly_degree_3 = 5
# curve_fit_3 = np.polyfit(x_axis, kids_values, poly_degree_3)
# poly_interp_3 = np.poly1d(curve_fit_3)
# poly_fit_values_3 = []

# for i in range(len(x_axis)):
#     poly_fit_values_1.append(poly_interp_1(i))
#     poly_fit_values_2.append(poly_interp_2(i))
#     poly_fit_values_3.append(poly_interp_3(i))

# plt.figure(figsize=(10.0, 10.0))
# plt.plot(x_axis, poly_fit_values_1, "o-y", label = "Fitted - 3rd order")
# plt.plot(x_axis, poly_fit_values_2, "o-r", label = "Fitted - 4th order")
# plt.plot(x_axis, poly_fit_values_3, "o-g", label = "Fitted - 5th order")
# plt.plot(x_axis, kids_values, "--", label = "Orig")

# plt.legend(loc="upper left")
# plt.show()
# plt.close()

x_axis2 = range(15)
poly_fit_values = []
# for i in range(len(x_axis2)):
#     poly_fit_values.append(poly_interp_3(i))

# plt.plot(x_axis2, poly_fit_values, "-g", label = "Prediction")
# plt.legend(loc="upper right")
# plt.show()
