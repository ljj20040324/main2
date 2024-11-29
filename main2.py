# 导入必要的库和模块
from sklearn.datasets import load_iris  # 导入加载iris数据集的函数
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯分类器类
from sklearn.metrics import accuracy_score  # 导入准确率计算函数
import pandas as pd  # 导入pandas用于更好地打印数据集

# 加载数据集
iris = load_iris()  # 加载iris数据集

# 打印Iris数据集的特征和目标标签
print("Iris数据集的特征:")
print(iris.data)  # 打印特征数据
print("\nIris数据集的目标标签:")
print(iris.target)  # 打印目标标签

# 获取特征数据和目标标签
X = iris.data
y = iris.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 划分数据集，30%作为测试集，随机种子为42

# 将数据集转换为DataFrame以便更好地打印
X_train_df = pd.DataFrame(X_train, columns=iris.feature_names)
X_test_df = pd.DataFrame(X_test, columns=iris.feature_names)
y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)

# 打印训练集和测试集的前几行
print("\n训练集特征变量 X_train 前5行:")
print(X_train_df.head())
print("\n训练集目标变量 y_train 前5个:")
print(y_train_series.head())
print("\n测试集特征变量 X_test 前5行:")
print(X_test_df.head())
print("\n测试集目标变量 y_test 前5个:")
print(y_test_series.head())

# 创建朴素贝叶斯分类器
gnb = GaussianNB()  # 创建高斯朴素贝叶斯分类器对象

# 训练模型
gnb.fit(X_train, y_train)  # 使用训练集训练朴素贝叶斯分类器

# 预测
y_pred = gnb.predict(X_test)  # 使用测试集进行预测

# 打印预测结果与真实分类结果
print("朴素贝叶斯分类器的预测结果与真实分类结果:")
for i, (pred, true) in enumerate(zip(y_pred, y_test)):
    print(f"样本 {i+1}: 预测类别 = {pred}, 真实类别 = {true}")

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)  # 计算预测结果的准确率
print(f'\n朴素贝叶斯分类器的准确率: {accuracy:.2f}')  # 打印准确率
