import matplotlib.pyplot as plt

depth = [1,2,3,4,5,6,7,8,9,10]
MSE_train = [257, 121, 75, 58, 44, 32, 23, 17, 12, 8]
MSE_val = [257, 127, 81, 68, 59, 54, 53, 54, 57, 59]

plt.plot(depth, MSE_train, label = 'MSE training data')
plt.plot(depth, MSE_val, label = "MSE validation data")
plt.xticks(depth)
plt.yticks([i for i in range(0,275,25)])

plt.xlabel("Depth of regression tree")
plt.ylabel("Mean squared error")
plt.legend()
plt.title('Depth of regression tree and mean squared error')
plt.show()