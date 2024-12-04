# 1. 导入数据
data <- read.csv("data.csv", fileEncoding = "UTF-8")

# 2. 选择相关的变量
# 选择v2到v11的列进行分析，v1是年份，不参与分析
selected_data <- data[, c("v1","v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11")]

# 3. 绘制散点图矩阵
pairs(selected_data, 
      main = "散点图矩阵 - 各指标与人口老龄化的关系", 
      pch = 19,           # 设置点的形状
      col = "blue",       # 设置点的颜色
)

# 4. 计算各指标与老龄化比例（v11）之间的相关性系数
correlation_matrix <- cor(selected_data)
print(correlation_matrix)

# 5. 根据相关系数筛选变量
# 筛选与老龄化比例（v11）相关性较强的变量
relevant_vars <- selected_data[, abs(correlation_matrix["v11", ]) > 0.3]

# 6. 绘制筛选后的相关变量的散点图矩阵
pairs(relevant_vars, 
      main = "筛选后的散点图矩阵 - 相关指标与人口老龄化的关系", 
      pch = 19, 
      col = "blue", 
)

