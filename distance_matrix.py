import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 假设20个点的坐标，每一行表示一个点的横坐标和纵坐标
points = np.array([[0.36247087, 0.7458513 ],
 [0.5511522,  0.09447861],
 [0.2682072,  0.65452135],
 [0.36780787, 0.17495584],
 [0.6831076,  0.7852808 ],
 [0.5596329,  0.94189596],
 [0.9827993,  0.45058942],
 [0.02969646, 0.71623313],
 [0.70745134, 0.04872954],
 [0.84489036, 0.86312807],
 [0.19697487, 0.7397634 ],
 [0.23373282, 0.42303085],
 [0.6353364,  0.48408413],
 [0.94960463, 0.09483778],
 [0.01898181, 0.6214701 ],
 [0.7316824,  0.6385814 ],
 [0.47450817, 0.44635797],
 [0.75009143, 0.13691413],
 [0.25756907, 0.61939895],
 [0.4334824,  0.8900689 ]])

#绘制坐标图
# Adding an additional red point to the scatter plot
additional_point = [0.9, 0.8]

plt.figure(figsize=(10, 6))

# Plot the original points
for i, point in enumerate(points):
    plt.scatter(point[0], point[1], c='b', marker='o')  # Blue color, circular marker
    plt.text(point[0] + 0.02, point[1] + 0.02, f'R{i+1}', fontsize=9, ha='right')

# Plot the additional point
plt.scatter(additional_point[0], additional_point[1], c='r', marker='o')  # Red color, circular marker
plt.text(additional_point[0] + 0.02, additional_point[1] + 0.02, 'Supplier', fontsize=9, ha='right')

plt.title('')
plt.xlabel('X 轴坐标')
plt.ylabel('Y 轴坐标')
plt.grid(True)
plt.show()


# 计算距离矩阵
distance_matrix = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)

# 将距离矩阵转为 Pandas DataFrame
df_distance = pd.DataFrame(distance_matrix, index=np.arange(1, 21), columns=np.arange(1, 21))

# 打印表格
#print("Distance Matrix:")
#print(df_distance)
# 将DataFrame保存为CSV文件
# 将DataFrame保存为Excel文件
df_distance.to_excel('distance_matrix_20.xlsx', index_label='Point')

points_50 = np.array([[0.30618334, 0.1478957 ],
        [0.84065044, 0.19573522],
        [0.60919523, 0.88604224],
        [0.76208353, 0.73309946],
        [0.52969384, 0.0506624 ],
        [0.66456914, 0.32229936],
        [0.31915045, 0.82504404],
        [0.65568626, 0.9197707 ],
        [0.07447338, 0.01342118],
        [0.10886848, 0.44396365],
        [0.88966   , 0.38419402],
        [0.58708465, 0.24229121],
        [0.74286854, 0.20632231],
        [0.17254686, 0.98633134],
        [0.2524588 , 0.5176436 ],
        [0.22884572, 0.2133205 ],
        [0.4062661 , 0.1572491 ],
        [0.13244736, 0.87542605],
        [0.9317936 , 0.63320184],
        [0.18568456, 0.9979353 ],
        [0.42231   , 0.25753176],
        [0.42521703, 0.32558405],
        [0.36174595, 0.8030225 ],
        [0.40440202, 0.5381404 ],
        [0.5987787 , 0.03128147],
        [0.39228296, 0.05941606],
        [0.40608466, 0.9002538 ],
        [0.4193977 , 0.335405  ],
        [0.5870552 , 0.19643128],
        [0.3296095 , 0.5888636 ],
        [0.3097639 , 0.31007612],
        [0.40439057, 0.84135365],
        [0.17406046, 0.60326064],
        [0.83258104, 0.42387807],
        [0.40761662, 0.5541574 ],
        [0.22670758, 0.29475915],
        [0.267689  , 0.92411935],
        [0.3804202 , 0.27074003],
        [0.74454975, 0.19275141],
        [0.33509326, 0.80709875],
        [0.6245662 , 0.64525986],
        [0.5035373 , 0.53130364],
        [0.6471896 , 0.93830776],
        [0.86892605, 0.38814116],
        [0.78176117, 0.24311125],
        [0.18490267, 0.1859889 ],
        [0.8494359 , 0.43954527],
        [0.13048267, 0.6464726 ],
        [0.18347049, 0.8314116 ],
        [0.7947569 , 0.49586797]])
# 计算距离矩阵
distance_matrix_50 = np.linalg.norm(points_50[:, np.newaxis, :] - points_50[np.newaxis, :, :], axis=-1)

# 将距离矩阵转为 Pandas DataFrame
df_distance_50 = pd.DataFrame(distance_matrix_50, index=np.arange(1, 51), columns=np.arange(1, 51))


df_distance_50.to_excel('distance_matrix_50.xlsx', index_label='Point')
