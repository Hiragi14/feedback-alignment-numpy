import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import japanize_matplotlib
# japanize_matplotlib.japanize()

df1 = pd.read_csv('./data/data_dfa_eye.csv')
df2 = pd.read_csv('./data/data_dfa_random.csv')
df3 = pd.read_csv('./data/data_dfa_ones.csv')
df4 = pd.read_csv('./data/data_fa.csv')
df5 = pd.read_csv('./data/data_dfa.csv')

Accuracy_BP = df5['Accuracy_BP']
Accuracy_FA = df4['Accuracy_FA']
Accuracy_DFA_random = df2['Accuracy_DFA']
Accuracy_DFA_ones = df3['Accuracy_DFA']
Accuracy_DFA_eye = df1['Accuracy_DFA']
Accuracy_DFA_tile = df5['Accuracy_DFA']

Cost_BP = df5['Cost_BP']
Cost_FA = df4['Cost_FA']
Cost_DFA_random = df2['Cost_DFA']
Cost_DFA_ones = df3['Cost_DFA']
Cost_DFA_eye = df1['Cost_DFA']
Cost_DFA_tile = df5['Cost_DFA']

Cos_sim_FA_L1 = df4['cos_sim_layer1']
Cos_sim_FA_L2 = df4['cos_sim_layer2']
Cos_sim_FA_L3 = df4['cos_sim_layer3']
Cos_sim_FA_ALL = df4['cos_sim_all']

Cos_sim_DFA_random_L1 = df2['cos_sim_layer1']
Cos_sim_DFA_random_L2 = df2['cos_sim_layer2']
Cos_sim_DFA_random_L3 = df2['cos_sim_layer3']
Cos_sim_DFA_random_ALL = df2['cos_sim_all']

Cos_sim_DFA_ones_L1 = df3['cos_sim_layer1']
Cos_sim_DFA_ones_L2 = df3['cos_sim_layer2']
Cos_sim_DFA_ones_L3 = df3['cos_sim_layer3']
Cos_sim_DFA_ones_ALL = df3['cos_sim_all']

Cos_sim_DFA_eye_L1 = df1['cos_sim_layer1']
Cos_sim_DFA_eye_L2 = df1['cos_sim_layer2']
Cos_sim_DFA_eye_L3 = df1['cos_sim_layer3']
Cos_sim_DFA_eye_ALL = df1['cos_sim_all']

epoch = 80
x1 = range(epoch)
x = range(50)

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
# ax.plot(x,Accuracy_DFA_random,'.',label='DFA(random-matrix)',linestyle="solid")
# ax.plot(x,Accuracy_DFA_ones,'.',label='DFA(ones-matrix)',linestyle="solid")
ax.plot(x,Accuracy_DFA_eye,'.',label='DFA(eye-matrix)',linestyle="solid")
ax.plot(x,Accuracy_DFA_tile,'.',label='DFA(tile)',linestyle="solid")

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
# ax2.plot(x,Cost_DFA_random,'.',label='DFA(random-matrix)',linestyle="solid")
# ax2.plot(x,Cost_DFA_ones,'.',label='DFA(ones-matrix)',linestyle="solid")
ax2.plot(x,Cost_DFA_eye,'.',label='DFA(eye-matrix)',linestyle="solid")
ax2.plot(x,Cost_DFA_tile,'.',label='DFA(tile)',linestyle="solid")

ax2.legend(loc = 'best')

plt.show()

# #######
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111)
# #ラベル指定
# plt.title('Cos-Similarity (Layer3) Comparisons')
# plt.xlabel('epoch')
# plt.ylabel('cos_similarity')

# ax.minorticks_on()
# ax.tick_params(length = 6, width=0.5, colors = "black")
# plt.grid(True,color='gray',alpha = 0.3,linestyle = "--")

# #メモリを乗数表記に
# plt.gca().get_xaxis().get_major_formatter().set_powerlimits([-3,3])

# ax.plot(x,Cos_sim_FA_L3,'.',label='FA',linestyle="solid")
# ax.plot(x,Cos_sim_DFA_random_L3,'.',label='DFA(random-matrix)',linestyle="solid")
# ax.plot(x,Cos_sim_DFA_ones_L3,'.',label='DFA(ones-matrix)',linestyle="solid")
# ax.plot(x,Cos_sim_DFA_eye_L3,'.',label='DFA(eye-matrix)',linestyle="solid")

# ax.legend(loc = 'best')
# plt.show()

fig.savefig('Accuracy_tile_05.png')
fig2.savefig('Cost_tile_05.png')