import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize, NonlinearConstraint

#######################
# 数据读取与准备
#######################
# 假设您的数据保存在 data2.csv 中，无表头，使用空格或制表符分隔。
# 若为制表符分隔，可尝试 sep='\t' 或 sep=None 让pandas自动判断
df = pd.read_csv("data2.csv", header=None, sep=None, engine='python')

# 指定列名
df.columns = ["Year","Pop_end","BR","DR","NGR","Age_0_14","Age_15_64","Male","Female","GDP","ResidentC","Healthcare","AgingRatio"]

df = df.sort_values("Year")
Year = df["Year"].values
Population_end = df["Pop_end"].values
BirthRate = df["BR"].values
DeathRate = df["DR"].values
NaturalGrowthRate = df["NGR"].values
Age_0_14 = df["Age_0_14"].values
Age_15_64 = df["Age_15_64"].values
MalePop = df["Male"].values
FemalePop = df["Female"].values
GDP_percapita = df["GDP"].values
ResidentConsumption = df["ResidentC"].values
HealthcareCost = df["Healthcare"].values
AgingRatio = df["AgingRatio"].values

last_year = Year[-1]
future_len = 7
pred_Year = np.arange(last_year+1, last_year+1+future_len)  # 2024-2030

#######################
# 简单预测函数(线性回归)
#######################
def linear_predict(x, y, future):
    x_resh = x.reshape(-1,1)
    model = LinearRegression()
    model.fit(x_resh,y)
    return model.predict(future.reshape(-1,1))

#######################
# 出生率两种预测方法
#######################
BR_min = BirthRate.min()
offset_br = 0
if BR_min <=0:
    offset_br = abs(BR_min)+0.001
log_BR = np.log(BirthRate+offset_br)
# 方法1：线性(对数变换)
p_log = np.polyfit(Year, log_BR,1)
pred_logBR = np.polyval(p_log, pred_Year)
pred_BR_linear = np.exp(pred_logBR)-offset_br
pred_BR_linear[pred_BR_linear<0.001]=0.001

# 方法2：ARIMA预测出生率
arima_model = pm.auto_arima(BirthRate+offset_br, seasonal=False, trace=False)
pred_BR_arima = arima_model.predict(n_periods=future_len)
pred_BR_arima = pred_BR_arima - offset_br
pred_BR_arima[pred_BR_arima<0.001]=0.001

#######################
# 其他变量用线性预测
#######################
pred_Pop = linear_predict(Year, Population_end, pred_Year)
pred_Dr = linear_predict(Year, DeathRate, pred_Year)
pred_Dr[pred_Dr<0.001]=0.001
pred_A0_14 = linear_predict(Year, Age_0_14, pred_Year)
pred_A15_64 = linear_predict(Year, Age_15_64, pred_Year)
pred_Aging = linear_predict(Year, AgingRatio, pred_Year)

male_ratio = MalePop[-1]/(MalePop[-1]+FemalePop[-1])
pred_Male = pred_Pop*male_ratio
pred_Female = pred_Pop*(1-male_ratio)

pred_GDP = linear_predict(Year, GDP_percapita, pred_Year)
pred_RC = linear_predict(Year, ResidentConsumption, pred_Year)
pred_HC = linear_predict(Year, HealthcareCost, pred_Year)

#######################
# 定义优化和约束
#######################
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

    (pred_Pop, pred_BR, pred_Dr, pred_A0_14, pred_A15_64, pred_Aging, pred_Male, pred_Female) = pred

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
    # 年龄结构合计=100%
    for i in range(future_len):
        eqs.append(A0_14a[i] + A15_64a[i] + AgingA[i]*100 - 100)

    # 男+女=人口
    for i in range(future_len):
        eqs.append(M[i]+F[i]-Pop[i])

    # 自然增长率匹配
    for i in range(1,future_len):
        ngr_calc = ((Pop[i]-Pop[i-1])/Pop[i-1])*1000
        ngr_model = BR[i-1]-DR[i-1]
        eqs.append(ngr_calc - ngr_model)

    return eqs

eq_cons = NonlinearConstraint(constraints_fun, lb=0, ub=0)

def pos_BR_DR(x):
    BR = x[future_len:2*future_len]
    DR = x[2*future_len:3*future_len]
    return np.concatenate([BR,DR])
pos_cons = NonlinearConstraint(pos_BR_DR, lb=[0.001]*(future_len*2), ub=[np.inf]*(future_len*2))

def run_optimization(pred_set):
    (pred_Pop, pred_BR, pred_Dr, pred_A0_14, pred_A15_64, pred_Aging, pred_Male, pred_Female) = pred_set
    x0 = pack_vars(pred_Pop, pred_BR, pred_Dr, pred_A0_14, pred_A15_64, pred_Aging, pred_Male, pred_Female)
    res = minimize(lambda x: objective(x, pred_set), x0, method='trust-constr', constraints=[eq_cons, pos_cons], options={'verbose':0})
    return res

# 准备两个方案的预测集
pred_set_linear = (pred_Pop, pred_BR_linear, pred_Dr, pred_A0_14, pred_A15_64, pred_Aging, pred_Male, pred_Female)
res_linear = run_optimization(pred_set_linear)

pred_set_arima = (pred_Pop, pred_BR_arima, pred_Dr, pred_A0_14, pred_A15_64, pred_Aging, pred_Male, pred_Female)
res_arima = run_optimization(pred_set_arima)

scenarios = []
for name, res, pred_set in [("Linear_BR", res_linear, pred_set_linear), ("ARIMA_BR", res_arima, pred_set_arima)]:
    (pPop, pBR, pDr, pA0_14, pA15_64, pAging, pM, pF) = pred_set
    if res.success:
        final_x = res.x
        Pop_final = final_x[0:future_len]
        BR_final = final_x[future_len:2*future_len]
        DR_final = final_x[2*future_len:3*future_len]
        A0_14_final = final_x[3*future_len:4*future_len]
        A15_64_final = final_x[4*future_len:5*future_len]
        Aging_final = final_x[5*future_len:6*future_len]
        M_final = final_x[6*future_len:7*future_len]
        F_final = final_x[7*future_len:8*future_len]

        cost = objective(final_x, pred_set)
        final_df = pd.DataFrame({
            "Year": pred_Year,
            "Population_end(万人)": Pop_final,
            "BirthRate(‰)": BR_final,
            "DeathRate(‰)": DR_final,
            "Age_0_14(%)": A0_14_final,
            "Age_15_64(%)": A15_64_final,
            "AgingRatio(%)": Aging_final*100,
            "MalePop(万人)": M_final,
            "FemalePop(万人)": F_final,
            "GDP_percapita(元)": pred_GDP,
            "ResidentConsumption(元)": pred_RC,
            "HealthcareCost(亿元)": pred_HC
        })
        outfile = f"final_optimized_result_{name}.csv"
        final_df.to_csv(outfile, index=False)
        scenarios.append((name, cost, outfile, res.success))
    else:
        scenarios.append((name, np.inf, "", res.success))

scenarios = sorted(scenarios, key=lambda x: x[1])
best_scenario = scenarios[0]

print("所有方案结果：")
for s in scenarios:
    print(f"方案:{s[0]}, 最终代价:{s[1]}, 成功:{s[3]}, 文件:{s[2]}")

print("理论最优解方案:", best_scenario[0], "代价:", best_scenario[1], "文件:", best_scenario[2])
