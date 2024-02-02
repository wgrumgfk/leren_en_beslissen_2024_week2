import matplotlib.pyplot as plt

depth = [1,2,3,4,5,6,7,8,9,10]
MSE_train = [256, 119, 76.3, 57.3, 39.8, 26.1, 16.1, 9.56, 5.67, 3.31]
MSE_val = [263, 124, 83.4, 67.2, 52.0, 39.9, 32.4, 27.3, 24.6, 23.1]

plt.plot(depth, MSE_train, label = 'MSE training data')
plt.plot(depth, MSE_val, label = "MSE validation data")
plt.xticks(depth)
plt.yticks([i for i in range(0,275,25)])

plt.xlabel("Depth of regression tree")
plt.ylabel("Mean squared error")
plt.legend()
plt.show()