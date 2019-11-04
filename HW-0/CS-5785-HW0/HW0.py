
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import numpy as np


# In[3]:


file = open("iris.data")
features = []
labels = []
while 1:
    line = file.readline().strip('\n')
    if not line:
        break
    tmp = line.split(',')
    features.append(tmp[:4])
    labels.append(tmp[4])
features = np.array(features)


# In[4]:


colors = []
for i in labels:
    if i == str("Iris-setosa"):
        colors.append(str("r"))
    elif i == str("Iris-versicolor"):
        colors.append(str("g"))
    else:
        colors.append(str("b"))


# In[7]:


feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
# plt.figure(0)
for x in range(0,3):
    for y in range(x+1,4):
        xs = features[:,x]
        ys = features[:,y]
#         plt.subplot(4,4,x*4+y+1)
        plt.scatter(xs, ys, c=colors)
        plt.xlabel(feature_names[x])
        plt.ylabel(feature_names[y])
        plt.title("Iris Setosa:red, Iris Versicolour:green, Iris Virginica:blue")
# plt.savefig("iris_all.png")
# plt.close()
        plt.savefig("iris_plot" + str(x) + str(y) + ".png")
        plt.close()

