import matplotlib.pyplot as plt
import numpy as np

# DATA SET
days = list(range(1, 9))
temp_max = [25, 24, 27, 27, 30, 32, 33, 29]
temp_min = [13, 16, 15, 16, 19, 21, 20, 18]
# DATA PLOT
plt.xlabel('day')
plt.ylabel('tempture')
plt.plot(days, temp_max, 'ro')
plt.plot(days, temp_min, 'bo')
xmin, xmax, ymin, ymax = 0, 10, 14, 45
plt.axis([xmin, xmax, ymin, ymax])
plt.show()


# DATA SET
x = np.linspace(0, 2*np.pi, 50)
y1 = 3 * np.sin(x)
y2 = np.sin(2*x)
y3 = 0.3 * np.sin(x)
# DATA PLOT
startx, endx = 0, 6.29
starty, endy = -3.1, 3.1
plt.axis([startx, endx, starty, endy])
plt.plot(x, y1, 'ro')
plt.plot(x, y2, 'bx')
plt.plot(x, y3, 'y+')
plt.show()

y4 = np.cos(x)
plt.plot(x, y4, color='blue', linewidth=2, linestyle='-')
plt.show()


# DATASET
x = np.linspace(0, 25, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# DATAPLOT
plt.plot(x, y1, '-b', label='sin')
plt.plot(x, y2, '-r', label='cos')
plt.legend(loc='uper left')    # 凡例：labelの値を受けて出力
plt.ylim(-1.5, 2.0)
plt.title('Sine & Cosine PLOT')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.grid(True)
plt.show()

# Function
def sq(i):
    returen i**2
def cube(i):
    returen i**3
def power4(i):
    returen i**4

plt.plot(x, sq(x), 'r')
plt.plot(x, cube(x), 'b')
plt.plot(x, power4(x), 'y')
plt.show()


# DATASET
v = np.linspace(-np.pi, np.pi, 100)
x1 = np.sin(2*v)
x2 = np.cos(2*v)

# SUBPLOT
fig, (Left, Right) = plt.subplots(ncol=2, figsize=(10, 10))

Left.plt(v, x1, linewidth=2)
Left.set_title('sin')
Left.set_xlabel('Value')
Left.set_ylabel('X')
Left.set_xlim(-np.pi, np.pi)
Left.grid(True)

Right.plt(v, x1, linewidth=2)
Right.set_title('cos')
Right.set_xlabel('Value')
Right.set_ylabel('X')
Right.set_xlim(-np.pi, np.pi)
Right.grid(True)

fig.show()
