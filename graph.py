# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import japanize_matplotlib
# japanize_matplotlib.japanize()

df1 = pd.read_csv('./data/data_fa.csv')

Accuracy_BP = df1['Accuracy_BP']
Accuracy_FA = df1['Accuracy_FA']

Cost_BP = df1['Cost_BP']
Cost_FA = df1['Cost_FA']

epoch = 50
x = range(epoch)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
#ラベル指定
plt.title('Accuracy Comparisons')
plt.xlabel('epoch')
plt.ylabel('Accuracy')

ax.minorticks_on()
ax.tick_params(length = 6, width=0.5, colors = "black")
plt.grid(True,color='gray',alpha = 0.3,linestyle = "--")

#メモリを乗数表記に
plt.gca().get_xaxis().get_major_formatter().set_powerlimits([-3,3])

ax.plot(x,Accuracy_BP,'.',label='BP',linestyle="solid")
ax.plot(x,Accuracy_FA,'.',label='FA',linestyle="solid")

ax.legend(loc = 'best')
plt.show()

########
fig2 = plt.figure(figsize=(6,6))
ax2 = fig2.add_subplot(111)
#ラベル指定
plt.title('Cost Comparisons')
plt.xlabel('epoch')
plt.ylabel('Cost')

ax2.minorticks_on()
ax2.tick_params(length = 6, width=0.5, colors = "black")
plt.grid(True,color='gray',alpha = 0.3,linestyle = "--")

#メモリを乗数表記に
plt.gca().get_xaxis().get_major_formatter().set_powerlimits([-3,3])

ax2.plot(x,Cost_BP,'.',label='BP',linestyle="solid")
ax2.plot(x,Cost_FA,'.',label='FA',linestyle="solid")

ax2.legend(loc = 'best')

plt.show()

fig.savefig('./image/Accuracy.png')
fig2.savefig('./image/Cost.png')