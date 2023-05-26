import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# 创建虚拟数据
data = pd.DataFrame({
    'feature1': np.random.rand(1000),
    'feature2': np.random.rand(1000),
    'feature3': np.random.rand(1000),
    'clicked': np.random.randint(0, 2, 1000)
})

# 定义特征和目标变量
X = data[['feature1', 'feature2', 'feature3']]
y = data['clicked']

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型
log_reg = LogisticRegression()

# 训练模型
log_reg.fit(X_train, y_train)

# 预测测试集
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# 计算准确率和AUC
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("Accuracy: ", accuracy)
print("AUC: ", auc)