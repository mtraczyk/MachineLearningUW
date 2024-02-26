import matplotlib.pyplot as plt

# Read data from command line arguments
# The first argument is the data to plot

data_acc = [float(x) for x in open("data_acc.txt").read().split(",")]

plt.plot(data_acc)
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.title("Acceleration over time")
plt.show()


data_att = [float(x) for x in open("data_att.txt").read().split(",")]

plt.plot(data_att)
plt.xlabel("Time")
plt.ylabel("Altitude")
plt.title("Altitude over time")
plt.show()
