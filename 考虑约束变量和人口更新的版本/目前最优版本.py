import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize, NonlinearConstraint

# -------------------- 数据读取与准备 --------------------
df = pd.read_csv("data2.csv", header=None, sep=None, engine='python')
df.columns = ["Year", "Pop_end", "BR", "DR", "NGR", "Age_0_14", "Age_15_64",
              "Male", "Female", "GDP", "ResidentC", "Healthcare", "AgingRatio"]
df = df.sort_values("Year")

Year = df["Year"].values
Population_end = df["Pop_end"].values
BirthRate = df["BR"].values
DeathRate = df["DR"].values
Age_0_14 = df["Age_0_14"].values
Age_15_64 = df["Age_15_64"].values
AgingRatio = df["AgingRatio"].values
MalePop = df["Male"].values
FemalePop = df["Female"].values
GDP_percapita = df["GDP"].values
ResidentConsumption = df["ResidentC"].values
HealthcareCost = df["Healthcare"].values

last_year = Year[-1]
future_years = np.arange(last_year + 1, 2051)
future_len = len(future_years)
pred_Year = future_years

# -------------------- 灰色预测模型 GM(1,1) --------------------
def gm11(y):
    n = len(y)
    x1 = np.cumsum(y)  # 累加序列
    z = (x1[:-1] + x1[1:]) / 2  # 紧邻均值生成序列
    B = np.vstack([-z, np.ones(n - 1)]).T
    Yn = y[1:]
    u = np.linalg.inv(B.T @ B) @ B.T @ Yn  # 求解参数
    a, b = u
    f = lambda k: (y[0] - b / a) * np.exp(-a * k) + b / a  # 预测公式
    return np.array([f(i) for i in range(len(y) + future_len)]), a

gm_prediction, gm_a = gm11(BirthRate)
pred_BR_gm = gm_prediction[:future_len]

# -------------------- 对数线性预测 --------------------
BR_offset = max(0, -BirthRate.min() + 0.001)
log_BR = np.log(BirthRate + BR_offset)
p_log = np.polyfit(Year, log_BR, 1)
pred_logBR = np.polyval(p_log, pred_Year)
pred_BR_linear = np.exp(pred_logBR) - BR_offset
pred_BR_linear = np.maximum(pred_BR_linear, 0.001)

# -------------------- 组合预测 --------------------
pred_BR_combined = 0.5 * pred_BR_gm + 0.5 * pred_BR_linear

# -------------------- 年龄段比例和老龄化比例预测 --------------------
def linear_predict(x, y, future):
    x_reshaped = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_reshaped, y)
    return model.predict(future.reshape(-1, 1))

pred_A0_14 = linear_predict(Year, Age_0_14, pred_Year)
pred_A15_64 = linear_predict(Year, Age_15_64, pred_Year)
pred_Aging = linear_predict(Year, AgingRatio, pred_Year)

# -------------------- 动态更新人口结构 --------------------
life_expectancy = 80
pred_Dr = np.full(future_len, 1 / life_expectancy * 1000)
Pop_future = [Population_end[-1]]

for i in range(future_len):
    next_pop = Pop_future[-1] + (pred_BR_combined[i] - pred_Dr[i]) / 1000 * Pop_future[-1]
    Pop_future.append(next_pop)
Pop_future = np.array(Pop_future[1:])

male_ratio = MalePop[-1] / (MalePop[-1] + FemalePop[-1])
pred_Male = Pop_future * male_ratio
pred_Female = Pop_future * (1 - male_ratio)

# -------------------- 优化模型设置 --------------------
def pack_vars(Pop, BR, DR, A0_14, A15_64, Aging, M, F):
    return np.hstack([Pop, BR, DR, A0_14, A15_64, Aging, M, F])

def objective(x, pred):
    Pop = x[0:future_len]
    BR = x[future_len:2*future_len]
    DR = x[2*future_len:3*future_len]
    A0_14a = x[3*future_len:4*future_len]
    A15_64a = x[4*future_len:5*future_len]
    AgingA = x[5*future_len:6*future_len]
    M = x[6*future_len:7*future_len]
    F = x[7*future_len:8*future_len]
    pred_Pop, pred_BR, pred_Dr, pred_A0_14, pred_A15_64, pred_Aging, pred_Male, pred_Female = pred
    cost = ((Pop - pred_Pop)**2).sum() + ((BR - pred_BR)**2).sum() + ((DR - pred_Dr)**2).sum()
    cost += ((AgingA - 0.154) < 0).sum() * 20  # 惩罚老龄化比例
    return cost

# -------------------- 约束条件 --------------------
def constraints_fun(x):
    Pop = x[0:future_len]
    BR = x[future_len:2*future_len]
    DR = x[2*future_len:3*future_len]
    eqs = []
    for i in range(1, future_len):
        expected_pop = Pop[i-1] + (BR[i-1] - DR[i-1]) / 1000 * Pop[i-1]
        eqs.append(Pop[i] - expected_pop)
    return eqs

def aging_constraint(x):
    AgingA = x[5*future_len:6*future_len]
    return AgingA - 0.154  # AgingA >= 0.154

eq_cons = NonlinearConstraint(constraints_fun, lb=0, ub=0)
aging_cons = NonlinearConstraint(aging_constraint, lb=0, ub=np.inf)

x0 = pack_vars(Pop_future, pred_BR_combined, pred_Dr, pred_A0_14, pred_A15_64, pred_Aging, pred_Male, pred_Female)

# -------------------- 定义 pred_set --------------------
pred_set = (Pop_future, pred_BR_combined, pred_Dr, pred_A0_14, pred_A15_64, pred_Aging, pred_Male, pred_Female)

# 调用优化函数
res = minimize(lambda x: objective(x, pred_set), x0, method='trust-constr', constraints=[eq_cons, aging_cons], options={'verbose': 0})

# -------------------- 结果可视化与保存 --------------------
final_x = res.x
Pop_final = final_x[0:future_len]

plt.plot(pred_Year, Pop_final, marker='o')
plt.title("Population Predicted")
plt.xlabel("Year")
plt.ylabel("Population")
plt.grid(True)
plt.show()

result_df = pd.DataFrame({
    "Year": pred_Year,
    "Population_end": Pop_final
})

result_df.to_csv("optimized_prediction_combined4.csv", index=False)
print("预测结果已保存为 optimized_prediction_combined5.csv")
