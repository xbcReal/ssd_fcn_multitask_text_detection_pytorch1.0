# import numpy as np
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(6,4))
# ax = fig.add_subplot(111)
# x = np.linspace(-10,10,1000)
# y = np.where(x<0,0,x)#满足条件(condition)，输出x，不满足输出y
# y1 = np.where(x<=0,0,1)
# plt.xlim(-11,11)
# plt.ylim(-11,11)
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))
#
# plt.plot(x,y,label='ReLU',linestyle="-", color="blue")#label为标签
# plt.plot(x,y1,label='Deriv.ReLU',linestyle="--", color="red")#label为标签
# plt.legend(['ReLU','Deriv.ReLU'])
# plt.show()
# plt.savefig('ReLU.png', dpi=500) #指定分辨


#!/usr/bin/python #encoding:utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

x = np.arange(0,100000,5000)
a=np.array(x)
y1=[0,0.24,0.33,0.45,0.66,0.70,0.71,0.70,0.70,0.71,0.71,0.71,0.69,0.70,0.71,0.71,0.72,0.71,0.72,0.72]
y2=[0,0.33,0.45,0.56,0.70,0.74,0.76,0.77,0.82,0.81,0.82,0.80,0.81,0.82,0.83,0.84,0.85,0.87,0.86,0.87]
y3=[0,0.25,0.33,0.47,0.69,0.70,0.71,0.72,0.70,0.72,0.74,0.75,0.76,0.75,0.75,0.76,0.76,0.77,0.76,0.77]
y4=[0,0.33,0.45,0.54,0.68,0.73,0.76,0.75,0.81,0.82,0.80,0.78,0.80,0.80,0.82,0.83,0.84,0.83,0.84,0.84]


font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}
plt.xlabel('iteration times',font2)
plt.ylabel('F1 score',font2)
plt.xlim(0,100000)
plt.ylim(0,1)
ax = plt.gca()
xmajorLocator=MultipleLocator(10000) #将x主刻度标签设置为20的倍数
ax.xaxis.set_major_locator(xmajorLocator)
xminorLocator  = MultipleLocator(5000) #将x轴次刻度标签设置为5的倍数
ax.xaxis.set_minor_locator(xminorLocator)
ymajorLocator=MultipleLocator(0.1) #将x主刻度标签设置为20的倍数
ax.yaxis.set_major_locator(ymajorLocator)
yminorLocator  = MultipleLocator(0.01) #将x轴次刻度标签设置为5的倍数
ax.yaxis.set_minor_locator(yminorLocator)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
plt.plot(x,y2,label='ICDAR-2015 train dataset F1 score without multitask',linestyle="-", color="blue")#l
plt.plot(x,y1,label='ICDAR-2015 test dataset F1 score without multitask',linestyle="-", color="red")#label为标签
plt.plot(x,y4,label='ICDAR-2015 train dataset F1 score with multitask',linestyle="-", color="green")#label为标签
plt.plot(x,y3,label='ICDAR-2015 test dataset F1 score with multitask',linestyle="-", color="orange")#l

#plt.legend(loc=0,ncol=2)
plt.legend(['ICDAR-2015 train dataset F1 score without multitask','ICDAR-2015  test dataset F1 score without multitask',
            'ICDAR-2015 train dataset F1 score with multitask','ICDAR-2015 test dataset F1 score with multitask'],bbox_to_anchor=(0.2, 0.3))
plt.show()
plt.savefig('plot_test.png', dpi=500) #指定分辨率
