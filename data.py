import pandas as pd
import matplotlib as plt

data  = pd.read_csv("C:\Users\estra\AppData\Local\Temp\ff94861e-9505-4c06-940b-58c60a926376_b2c420e2-1ab2-4112-a8c4-2c0f4235fe63.zip.376\titanic.csv")
df = pd.DataFrame(data)
x_axis = list(df.iloc[:,0])
y_axis = list(df.iloc[:,1])

plt.bar(x_axis, y_axis, color="g")
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.show()