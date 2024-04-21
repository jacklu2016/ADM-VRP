import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 指定目录路径
file_path = r"E:\BaiduNetdiskWorkspace\VMI+Transportation\实验结果.xlsx"

data = pd.read_excel(file_path, skiprows=1, nrows=4)

labels = data.iloc[:, 0]
values_20 = data.iloc[:, 1]
values_50 = data.iloc[:, 5]
values_80 = data.iloc[:, 9]
values_100 = data.iloc[:, 13]

x = np.arange(len(labels))  # x轴位置
# 颜色
colors = ['red', 'blue', 'purple', 'orange']
# 绘图
width = 0.2  # 柱状图宽度
fig, ax = plt.subplots()

rects1 = ax.bar(x - 3*width/2, values_20, width, label='INV_20', color=colors[0])
rects2 = ax.bar(x - width/2, values_50, width, label='INV_50', color=colors[1])
rects3 = ax.bar(x + width/2, values_80, width, label='INV_80', color=colors[2])
rects4 = ax.bar(x + 3*width/2, values_100, width, label='INV_100', color=colors[3])

# 添加标签、标题和图例
ax.set_ylabel('INV.Cost')
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 调整图例位置
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
# 显示柱状图
plt.tight_layout()
plt.show()