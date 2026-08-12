import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
height=[[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0]]
weight=[  8, 10 , 12, 14, 16, 18, 20]
plt.scatter(height,weight,color='blue')
plt.xlabel("height")
plt.ylabel("weight")
bmi=linear_model.LinearRegression()
bmi.fit(height,weight)
X_height=[[13.0]]
print(bmi.predict(X_height))
