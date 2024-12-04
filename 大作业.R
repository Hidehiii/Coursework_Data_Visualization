
# 1. 导入数据
data <- read.csv("data.csv", fileEncoding = "UTF-8")
#----------------------------------------------------------------------------------查看数据--------------------------------------------------------------------------------------------
View(data)
# 2. 选择相关的变量
# 选择v2到v11的列进行分析，v1是年份，不参与分析
selected_data <- data[, c("v1","v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11")]

#------------------------------------------------------------------------------- 绘制散点图矩阵--------------------------------------------------------------------
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


#---------------------------------------------------------------------------------------------建立线性回归模型与生成人口老龄化比例预测图和数据证明线性模型可行度----------------------------------------------------------------------------------------------
# 加载必要的库
library(ggplot2)

# 1. 数据清洗：删除不需要的列（V3, V4, V5）
data_clean <- data[, -c(3, 4, 5)]

# 2. 建立线性回归模型，使用V2, V6, V7, V8, V9, V10预测V11
fit <- lm(v11 ~ v2 + v6 + v7 + v8 + v9 + v10, data = data_clean)

# 3. 查看回归模型的详细结果
summary(fit)

# 4. 使用回归模型对原数据进行预测
predicted_v11 <- predict(fit, newdata = data_clean)

# 5. 将预测结果添加到原数据框中
data_clean$predicted_v11 <- predicted_v11

# 6. 绘制实际值与预测值的对比图
ggplot(data_clean, aes(x = v1)) + 
  geom_line(aes(y = v11, color = "实际值"), size = 1) +  # 实际值（V11）
  geom_line(aes(y = predicted_v11, color = "预测值"), size = 1, linetype = "dashed") +  # 预测值（predicted_v11）
  labs(title = "人口老龄化比例预测", 
       x = "年份", 
       y = "人口老龄化比例") + 
  scale_color_manual(values = c("实际值" = "blue", "预测值" = "red")) +  # 定义颜色
  theme_minimal() +
  scale_x_continuous(breaks = seq(min(data_clean$v1), max(data_clean$v1), by = 1), 
                     labels = as.character(seq(min(data_clean$v1), max(data_clean$v1), by = 1))) +  # 设置x轴每年显示
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # 设置x轴标签倾斜，防止重叠
#-----------------------------计算误差--------------------------------------


# 7. 计算预测误差：相对误差、MSE、RMSE 和 MAE

# 相对误差计算（每年）
relative_errors <- (data_clean$v11 - data_clean$predicted_v11) / data_clean$v11

# 计算相对误差的平均值
avg_relative_error <- mean(relative_errors, na.rm = TRUE)

# 均方误差（MSE）
mse <- mean((data_clean$v11 - data_clean$predicted_v11)^2)

# 均根误差（RMSE）
rmse <- sqrt(mse)

# 均绝对误差（MAE）
mae <- mean(abs(data_clean$v11 - data_clean$predicted_v11))

# 8. 输出误差计算结果
cat("平均相对误差：", avg_relative_error, "\n")
cat("均方误差（MSE）：", mse, "\n")
cat("均根误差（RMSE）：", rmse, "\n")
cat("均绝对误差（MAE）：", mae, "\n")
