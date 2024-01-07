import numpy as np
import matplotlib.pyplot as plt

# print(np.zeros(10))

# validation_dataset = utils.create_data_on_disk(20,10,is_save=False,filename='filepath',is_return=True)
# for ele in validation_dataset:
#     print(ele)
# print(validation_dataset)

# 模拟每个零售商的需求，30天为例
seed = 2345
# 设置均值和标准差（可以根据需要调整）
mean_value = 25
std_dev = 10

# 生成 30 个正态分布的随机数
random_numbers = np.random.normal(mean_value, std_dev, 30)

# 限制数值范围在 1 到 50 之间
random_numbers = np.clip(random_numbers, 1, 50).round(0)

# 打印结果
# print(random_numbers)
c_demand = random_numbers

inventory = 100
s = 20  # 安全库存
q = 40  # 补货量

replenishs = []
stockouts = []
inventorys = []
for i in range(c_demand.shape[0],):
    # 如果库存大于用户需求，则减库存，如果小于用户需求，则记一次缺货
    if inventory >= c_demand[i]:
        inventory -= c_demand[i]
        stockouts.append(0)
    else:
        stockouts.append(c_demand[i],)

    # 如果小于安全库存s，则发起数量q的补货
    if inventory <= s:
        replenishs.append(q)
        inventory += q
    else:
        replenishs.append(0)

    inventorys.append(inventory)

print(np.array2string(c_demand, separator=', ', precision=0, suppress_small=True))
print(inventorys)
print(replenishs)
print(stockouts)
print(np.sum(np.array(inventorys) * 2),0.05)


print('================')
demand = [27, 17, 43, 13, 14, 22, 17, 26, 20,  1, 39, 33, 37, 16,  28, 37, 22,
42, 21, 26, 37, 22, 29, 27, 24,  9, 20, 16, 18, 23],
income = np.sum(demand * 20)
algorithms_inv = ['S,Q', 'GA', 'DDPG', 'MT-PPO']
profits = [10668.29, 8504.35, 8777.36, 11777.99]
gap_inv = []
cost_inv = []
for i in range(len(profits)):
    cost_inv.append(income - profits[i])

for i in range(len(profits)):
    gap_inv.append(cost_inv[i] / cost_inv[0] - 1)

print(np.array(cost_inv).round(2))
print(np.array(gap_inv).round(4))

print('====================')
algorithms_vrp = ['HGS', 'LKH3', 'PPO', 'MT-PPO']
costs_vrp = [6.14, 5.93, 6.26, 6.07]
gaps_vrp = []
for i in range(4):
    gap = round(costs_vrp[i] / costs_vrp[0] - 1, 4)
    gaps_vrp.append(gap)

print(f'gaps_vrp cost:{gaps_vrp}')

times_vrp_20 = [5.32, 4.98 ,0.26, 0.41]
gaps_times_vrp_20 = []
for i in range(4):
    gap = round(times_vrp_20[i] / times_vrp_20[0] - 1, 4)
    gaps_times_vrp_20.append(gap)

print(f'gaps_times_vrp_20:{gaps_times_vrp_20}')

#归一化总成本
def z_score_normalize(data):
    """
    使用 Z-Score 归一化对数据进行归一化。

    Parameters:
    - data: numpy array, 输入的原始数据。

    Returns:
    - normalized_data: numpy array, 归一化后的数据。
    """
    mean_val = np.mean(data)
    std_val = np.std(data)

    # 使用 Z-Score 归一化
    normalized_data = (data - mean_val) / std_val

    return normalized_data


# 示例使用
data = np.array(
    [44, 33, 24, 15, 7, 3, 3, 7, 13, 23, 33, 43, 50, 51, 50, 44, 35, 23, 13, 6, 1, 3, 7, 13, 22, 34, 42, 50, 52, 48],)
normalized_data = z_score_normalize(data)
print(normalized_data)

# 示例使用
#data = np.array(
#    [44, 33, 24, 15, 7, 3, 3, 7, 13, 23, 33, 43, 50, 51, 50, 44, 35, 23, 13, 6, 1, 3, 7, 13, 22, 34, 42, 50, 52, 48],)
#normalized_data = normalize_data(data)
#print(normalized_data)

cost_inv = z_score_normalize(np.array(cost_inv))
costs_vrp = z_score_normalize(np.array(costs_vrp))

for i in range(3):
    for j in range(3):
        algorithm = algorithms_inv[i] + '+' + algorithms_vrp[j]
        cost_sum = cost_inv[i] + costs_vrp[j]
        print(f'algorithms:{algorithm},{round(cost_inv[i], 2)},{round(costs_vrp[j], 2)},sum cost:{round(cost_sum, 2)}')

print(f"MT-PPO:{cost_inv[3],},{costs_vrp[3],},sum cost:{cost_inv[3] + costs_vrp[3]}")

import pandas as pd
points_coords = np.array([[0.20582926,0.7242837 ],
 [0.69336665,0.18752003],
 [0.74941325,0.5543035 ],
 [0.26518416,0.89508927],
 [0.7716173 ,0.52577937],
 [0.14699876,0.44511163],
 [0.4149766 ,0.9011805 ],
 [0.4292277 ,0.5411267 ],
 [0.49294388,0.13799083],
 [0.10816205,0.322644  ],
 [0.72092235,0.9739976 ],
 [0.77951515,0.41185796],
 [0.27562046,0.69787145],
 [0.9447346 ,0.40684366],
 [0.7629787 ,0.47627723],
 [0.47067976,0.5864005 ],
 [0.81109464,0.7845818 ],
 [0.589448  ,0.39752102],
 [0.84803796,0.8010752 ],
 [0.6473237 ,0.6311959 ],])

points_coords_df = pd.DataFrame(points_coords, columns=['x', 'y'])
#points_coords = points_coords.T
points_coords_df.T.to_excel('vrp20_nodes_coords.xlsx', index=False)

# 计算距离矩阵
distance_matrix = np.linalg.norm(points_coords[:, np.newaxis, :] - points_coords[np.newaxis, :, :], axis=-1)

# 将距离矩阵转为 Pandas DataFrame
df_distance = pd.DataFrame(distance_matrix, index=np.arange(1, 21), columns=np.arange(1, 21))

df_distance.to_excel('distance_matrix_20.xlsx', index=True)

#绘制地理位置坐标图
# 供应商坐标
red_point = np.array([0.9496393, 0.14876258])

# 绘制零售商散点图
plt.scatter(points_coords[:, 0], points_coords[:, 1], label='Retailer')
plt.scatter(red_point[0], red_point[1], color='red', label='Vendor')
# 显示图例
plt.legend()

plt.show()

#绘制零售商用户需求折线图
demand = [27, 17, 43, 13, 14, 22, 17, 26, 20,  1, 39, 33, 37, 16,  28, 37, 22,
42, 21, 26, 37, 22, 29, 27, 24,  9, 20, 16, 18, 23]

plt.plot(range(30),demand,)
plt.show()