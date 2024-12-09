import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
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

# -------------------- 线性预测函数 --------------------
def linear_predict(x, y, future):
    x_reshaped = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_reshaped, y)
    return model.predict(future.reshape(-1, 1)), model

# -------------------- 样条回归函数 --------------------
def spline_predict(x, y, future):
    model = make_pipeline(SplineTransformer(degree=3, n_knots=4), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    return model.predict(future.reshape(-1, 1)), model

# -------------------- 出生率、死亡率（取对数后）预测 --------------------
BR_offset = max(0, -BirthRate.min() + 0.001)
DR_offset = max(0, -DeathRate.min() + 0.001)

log_BR = np.log(BirthRate + BR_offset)
log_DR = np.log(DeathRate + DR_offset)

pred_log_BR, log_BR_model = spline_predict(Year, log_BR, pred_Year)
pred_log_DR, log_DR_model = spline_predict(Year, log_DR, pred_Year)

pred_BR = np.exp(pred_log_BR) - BR_offset
pred_DR = np.exp(pred_log_DR) - DR_offset
pred_NGR = pred_BR - pred_DR

# -------------------- 动态更新人口结构 --------------------
life_expectancy = 80
pred_Dr = np.full(future_len, 1 / life_expectancy * 1000)  # 假设固定死亡率

Pop_future = [Population_end[-1]]  # 使用基线年的人口作为起点
for i in range(future_len):
    next_pop = Pop_future[-1] + (pred_BR[i] - pred_Dr[i]) / 1000 * Pop_future[-1]
    Pop_future.append(next_pop)
Pop_future = np.array(Pop_future[1:])  # 去掉基线年的值

male_ratio = MalePop[-1] / (MalePop[-1] + FemalePop[-1])
pred_Male = Pop_future * male_ratio
pred_Female = Pop_future * (1 - male_ratio)

# -------------------- 年龄段比例和经济变量预测 --------------------
pred_A0_14, A0_14_model = linear_predict(Year, Age_0_14, pred_Year)
pred_A15_64, A15_64_model = linear_predict(Year, Age_15_64, pred_Year)
pred_GDP, GDP_model = linear_predict(Year, GDP_percapita, pred_Year)
pred_RC, RC_model = linear_predict(Year, ResidentConsumption, pred_Year)
pred_HC, HC_model = linear_predict(Year, HealthcareCost, pred_Year)

# -------------------- 老龄化比例预测 --------------------
X_features = np.column_stack([Age_0_14, Age_15_64, GDP_percapita, ResidentConsumption, HealthcareCost])
model_aging = RandomForestRegressor(n_estimators=100, random_state=42)
model_aging.fit(X_features, AgingRatio)
X_future = np.column_stack([pred_A0_14, pred_A15_64, pred_GDP, pred_RC, pred_HC])
pred_Aging = model_aging.predict(X_future)

# -------------------- 优化设置 --------------------
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
    cost = ((Pop - pred_Pop)**2).sum() + ((BR - pred_BR)**2).sum() + ((DR - pred_Dr)**2).sum() \
           + ((A0_14a - pred_A0_14)**2).sum() + ((A15_64a - pred_A15_64)**2).sum() + ((AgingA - pred_Aging)**2).sum() \
           + ((M - pred_Male)**2).sum() + ((F - pred_Female)**2).sum()
    return cost

def constraints_fun(x):
    Pop = x[0:future_len]
    BR = x[future_len:2*future_len]
    DR = x[2*future_len:3*future_len]
    A0_14a = x[3*future_len:4*future_len]
    A15_64a = x[4*future_len:5*future_len]
    AgingA = x[5*future_len:6*future_len]
    M = x[6*future_len:7*future_len]
    F = x[7*future_len:8*future_len]
    eqs = []
    for i in range(future_len):
        eqs.append(A0_14a[i] + A15_64a[i] + AgingA[i] * 100 - 100)
        eqs.append(M[i] + F[i] - Pop[i])
    for i in range(1, future_len):
        expected_pop = Pop[i-1] + (BR[i-1] - DR[i-1]) / 1000 * Pop[i-1]
        eqs.append(Pop[i] - expected_pop)
    return eqs

def aging_constraint(x):
    AgingA = x[5*future_len:6*future_len]
    return AgingA - 0.154

aging_cons = NonlinearConstraint(aging_constraint, lb=0, ub=np.inf)
eq_cons = NonlinearConstraint(constraints_fun, lb=0, ub=0)

x0 = pack_vars(Pop_future, pred_BR, pred_DR, pred_A0_14, pred_A15_64, pred_Aging, pred_Male, pred_Female)
pred_set = (Pop_future, pred_BR, pred_DR, pred_A0_14, pred_A15_64, pred_Aging, pred_Male, pred_Female)

res = minimize(lambda x: objective(x, pred_set), x0, method='trust-constr', constraints=[eq_cons, aging_cons])

# -------------------- 结果展示与保存 --------------------
final_x = res.x
Pop_final = final_x[0:future_len]
BR_final = final_x[future_len:2*future_len]
DR_final = final_x[2*future_len:3*future_len]
A0_14_final = final_x[3*future_len:4*future_len]
A15_64_final = final_x[4*future_len:5*future_len]
Aging_final = final_x[5*future_len:6*future_len]
M_final = final_x[6*future_len:7*future_len]
F_final = final_x[7*future_len:8*future_len]

variables = {
    "Population (Predicted)": Pop_final,
    "Birth Rate (Predicted)": BR_final,
    "Death Rate (Predicted)": DR_final,
    "Age 0-14 (%) (Predicted)": A0_14_final,
    "Age 15-64 (%) (Predicted)": A15_64_final,
    "Aging Ratio (%) (Predicted)": Aging_final * 100,
    "Male Population (Predicted)": M_final,
    "Female Population (Predicted)": F_final,
    "GDP per Capita (Predicted)": pred_GDP,
    "Resident Consumption (Predicted)": pred_RC,
    "Healthcare Cost (Predicted)": pred_HC
}

for title, data in variables.items():
    plt.figure(figsize=(10, 6))
    plt.plot(pred_Year, data, marker='o')
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(title)
    plt.grid(True)
    plt.show()

result_df = pd.DataFrame({
    "Year": pred_Year,
    "Population_end": Pop_final,
    "BirthRate": BR_final,
    "DeathRate": DR_final,
    "Age_0_14": A0_14_final,
    "Age_15_64": A15_64_final,
    "AgingRatio": Aging_final * 100,
    "MalePop": M_final,
    "FemalePop": F_final,
    "GDP_percapita": pred_GDP,
    "ResidentConsumption": pred_RC,
    "HealthcareCost": pred_HC
})

result_df.to_csv("optimized_prediction_results12.csv", index=False)
print("预测结果已保存为 optimized_prediction_results.csv")
