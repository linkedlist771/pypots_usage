import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.imputation import SAITS
from pypots.utils.metrics import calc_mae

# 生成一维时间序列数据
def generate_time_series(n_samples, n_steps):
    series = []
    for _ in range(n_samples):
        time = np.linspace(0, 1, n_steps)
        phase_shift = np.random.uniform(0, 2 * np.pi)  # 随机相位偏移
        frequency = np.random.uniform(0.8, 1.2)  # 随机频率偏移
        noise = np.random.normal(0, 0.1, n_steps)  # 随机噪声
        sample_series = np.sin(2 * np.pi * frequency * time + phase_shift) + noise
        series.append(sample_series)
    return np.array(series)

# 数据预处理
n_samples, n_steps = 1000, 48
X = generate_time_series(n_samples, n_steps)
X_ori_un_standardized = X.copy()  # 保留原始数据用于验证

X = StandardScaler().fit_transform(X)
X = X.reshape(n_samples, n_steps, 1)  # 重塑为 (样本数, 时间步, 特征数)
X_ori = X.copy()  # 保留原始数据用于验证

# 添加缺失值
X = mcar(X, 0.1)  # 随机将10%的观测值设为缺失
dataset = {"X": X}  # 为模型创建输入数据集

print(f"数据形状: {X.shape}")  # 应该是 (1000, 48, 1)

# 模型训练
saits = SAITS(n_steps=n_steps, n_features=1, n_layers=2,     d_model=128,  # Adjusted from 64 to 128

              n_heads=4, d_k=32, d_v=32, d_ffn=64, dropout=0.1, epochs=300)

saits.fit(dataset)  # 在数据集上训练模型

# 数据插补
imputation = saits.impute(dataset)

# 计算插补误差
indicating_mask = np.isnan(X) ^ np.isnan(X_ori)
mae = calc_mae(imputation, np.nan_to_num(X_ori), indicating_mask)

print(f"平均绝对误差 (MAE): {mae}")

# 可选：保存和加载模型
# saits.save("save_it_here/saits_1d_timeseries.pypots")
# saits.load("save_it_here/saits_1d_timeseries.pypots")

# 可视化结果（选择第一个样本进行展示）
import matplotlib.pyplot as plt

# 对插补后的数据进行逆标准化处理
imputation_un_standardized = imputation.reshape(n_samples, n_steps)
imputation_un_standardized = StandardScaler().fit(X_ori_un_standardized).inverse_transform(imputation_un_standardized)

# 可视化结果（选择5个样本进行展示）
# 可视化结果（选择5个样本进行展示）
n_samples_to_plot = 5
sample_indices = np.random.choice(np.arange(n_samples), n_samples_to_plot, replace=False)

plt.figure(figsize=(15, 12))  # 增加图像的整体尺寸，使得子图更大更清晰

for i, sample_idx in enumerate(sample_indices):
    plt.subplot(n_samples_to_plot, 1, i + 1)
    plt.plot(X_ori_un_standardized[sample_idx, :], label='Original', color='blue', linewidth=2)  # 加粗线条
    plt.scatter(np.arange(n_steps)[np.isnan(X[sample_idx, :, 0])],
                X_ori_un_standardized[sample_idx, np.isnan(X[sample_idx, :, 0])],
                color='red', label='Missing', marker='x', s=100)  # 增大标记点的大小
    plt.plot(imputation_un_standardized[sample_idx, :], label='Imputed', color='green', linestyle='--', linewidth=2)  # 采用虚线，并加粗
    plt.title(f'Sample {sample_idx + 1}', fontsize=14)  # 增大标题字体
    plt.xlabel('Time Step', fontsize=12)  # 增大标签字体
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=10)  # 增大图例字体

plt.tight_layout()
plt.show()