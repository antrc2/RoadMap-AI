import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
# Tạo dữ liệu giả lập cho bài toán dự đoán giá nhà
data = {
    "area": [850, 900, 1000, 1100, 1500, 2000, 2200, 2500, 3000, 3500],
    "rooms": [2, 3, 3, 4, 4, 5, 4, 5, 6, 7],
    "floors": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
    "price": [500000, 550000, 600000, 650000, 700000, 800000, 850000, 900000, 950000, 1000000]
}

# Tạo DataFrame từ dữ liệu trên
df = pd.DataFrame(data)

# Chia thành đặc trưng (X) và nhãn (y)
X = df.drop("price", axis=1)
y = df["price"]

# Chia thành train và test (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeRegressor()

model.fit(np.array(X_train),np.array(y_train))

print(model.predict(X_test))
print(y_test)
