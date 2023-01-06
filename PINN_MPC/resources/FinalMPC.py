import ndjson
import pandas as pd
import os
import math
import numpy as np
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from casadi import *

ConfPredRadius = [0.0316, 0.0583, 0.1118, 0.1676, 0.2059, 0.25, 0.2816, 0.3036, 0.3353, 0.3574, 0.3795, 0.3985, 0.4206, 0.4427, 0.4617, 0.4838, 0.53, 0.5852, 0.6351, 0.6768]

Xped1 = [1.65, 1.73,	1.73, 1.71,	1.78,	1.76,	1.72,	1.74,	1.68,	1.66,	1.68,	1.58,	1.49,	1.35,	1.09,	0.93,	0.75,	0.51,	0.46,	0.41,	0.33]
Yped1 = [2.7, 2.09,	1.52,	0.91,	0.3,	-0.31,	-0.89,	-1.47,	-2.07,	-2.67,	-3.29,	-3.91,	-4.55,	-5.17,	-5.74,	-6.33,	-6.88,	-7.43,	-8.02,	-8.57,	-9.13]

Xped2 = [1.98, 1.96, 1.98, 1.86, 1.83, 1.86, 1.8, 1.81, 1.85, 1.84, 1.86, 1.9, 1.86, 1.9, 1.92, 1.87, 1.95, 1.99, 1.96, 2.07]
Yped2 = [3.74, 3.44, 3.11, 2.78, 2.43, 2.07, 1.7, 1.32, 0.94, 0.48, 0.01, -0.5, -0.92, -1.47, -1.94, -2.41, -2.91, -3.4, -3.89, -4.38]


eps = 0.2
T = 10
Delta = 1/10
L = 1
xref = 2.5
yref = -7.5

x_state = [0.5]*(T-1)
y_state = [-2.5]*(T-1)

opti = casadi.Opti()
x = opti.variable(T)
y = opti.variable(T)
V = opti.variable(T)
theta = opti.variable(T)

uv = opti.variable(T)
delta = opti.variable(T)

opti.set_initial(x[0], x_state[0])
opti.set_initial(y[0], y_state[0])
opti.set_initial(V[0], 2.5)
opti.set_initial(theta[0],-pi/8)

p_opts = {"expand":True}
s_opts = {"max_iter": 4000}
opti.solver('ipopt', p_opts, s_opts)

obj = 0
for i in range(1, T):
    #opti.subject_to(opti.bounded(-3, V[i], 3))
    opti.subject_to(opti.bounded(-pi/4, delta[i], pi/4))
    #opti.subject_to(opti.bounded(-3, uv[i], 3))
    opti.subject_to(x[i] == x[i-1] + Delta * (V[i-1]*cos(theta[i-1])))
    opti.subject_to(y[i] == y[i-1] + Delta * (V[i-1]*sin(theta[i-1])))
    opti.subject_to(theta[i] == theta[i-1] + Delta * (V[i-1]/L*tan(delta[i-1])))
    opti.subject_to(V[i] == V[i-1] + Delta*(uv[i-1]))
    
for i in range(T):
    obj += delta[i-1]**2 + uv[i-1]**2
    opti.minimize(obj)
    opti.subject_to(sqrt((x[i]-Xped1[i+T])**2+(y[i]-Yped1[i+T])**2) >= ConfPredRadius[i]+eps)
    opti.subject_to(sqrt((x[i]-Xped2[i+T])**2+(y[i]-Yped2[i+T])**2) >= ConfPredRadius[i]+eps)
    
opti.subject_to(x[0]-x_state[0] == 0)
opti.subject_to(y[0]-y_state[0] == 0)
opti.subject_to(x[T-1]-xref == 0)
opti.subject_to(y[T-1]-yref == 0)

sol = opti.solve()
uv_solved = sol.value(uv)
delta_solved = sol.value(delta)

xs = sol.value(x)
ys = sol.value(y)
vs = sol.value(V)
thetas = sol.value(theta)

Data = np.zeros((10,11))

for i in range(10):
    Data[i,0] = i
    Data[i,1] = x_state[0]
    Data[i,2] = y_state[0]
    Data[i,3] = vs[0]
    Data[i,4] = thetas[0]
    Data[i,5] = uv_solved[i]
    Data[i,6] = delta_solved[i]
    Data[i,7] = xs[i]
    Data[i,8] = ys[i]
    Data[i,9] = vs[i]
    Data[i,10] = thetas[i]

time = [0,1,2,3,4,5,6,7,8,9]
plt.subplot(2, 1, 1)
plt.plot(time, uv_solved, 'o-')
plt.title('Control Scheme')
plt.ylabel('Acceleration')

plt.subplot(2, 1, 2)
plt.plot(time, delta_solved, '.-')
plt.xlabel('time (s)')
plt.ylabel('Steering angle')

plt.show()
#import pandas as pd
#df = pd.DataFrame(data=Data)
#df.to_csv('out.csv', mode='a', index=False, header=False)
x_state = x_state + list(sol.value(x)) + [list(sol.value(x))[T-1]]
y_state = y_state + list(sol.value(y)) + [list(sol.value(y))[T-1]]

fig = plt.figure(dpi=100)
fig.set_figwidth(10)
fig.set_figheight(10)
axis = plt.axes(xlim =(-3, 6),
            ylim =(-9, 3))
ts = np.linspace(0,2*pi,100)

p1_line, = axis.plot([], [], '*', color = 'g', markersize = 15)
r_line, = axis.plot([], [], 'v', color = 'y', markersize = 12)
c1_line, = axis.plot([], [], '-', color = 'g')
af1_line, = axis.plot([], [], '*', color = 'k')
ch1_line, = axis.plot([], [], color = 'k')

p2_line, = axis.plot([], [], '*', color = 'b', markersize = 15)
c2_line, = axis.plot([], [], '-', color = 'b')
af2_line, = axis.plot([], [], '*', color = 'r')
ch2_line, = axis.plot([], [], color = 'r')

x_robot, y_robot = [], []
x_pred1, y_pred1 = [], []
x_pred2, y_pred2 = [], []
x1_circle, y1_circle = [], []
x2_circle, y2_circle = [], []
x_af1, y_af1 = [], []
x_af2, y_af2 = [], []
#arrow = axis.annotate("", xy=(1.65, 2.7), xytext=(0, 0), arrowprops={"facecolor": "black"})

original_x_1 = Xped1
original_y_1 = Yped1

original_x_2 = Xped2
original_y_2 = Yped2

def init():
  p1_line.set_data([], [])
  p2_line.set_data([], [])
  r_line.set_data([], [])
  c1_line.set_data([], [])
  c2_line.set_data([],[])
  af1_line.set_data([], [])
  af2_line.set_data([], [])
  ch1_line.set_data([], [])
  ch2_line.set_data([], [])
  return p1_line, r_line, c1_line, af1_line, ch1_line, p2_line, c2_line, af2_line, ch2_line

def animate(i):
  global arrow
  if i <= 9:
      x_r = x_state[i]
      y_r = y_state[i]
      x_robot.append(x_r)
      y_robot.append(y_r)
      r_line.set_data(x_robot, y_robot)

  if i >= 10:
      arrow = axis.annotate("", xy=(0, 2), xytext=(1, 1), arrowprops={"facecolor": "black"})
      center = np.array([x_state[i], y_state[i]])
      arrow.xy = center + ((x_state[i] - x_state[i-1])/5, (y_state[i] - y_state[i-1])/5)
      arrow.set_position(center)

  x_pred1, y_pred1 = [], []
  x_p1 = original_x_1[i]
  y_p1 = original_y_1[i]

  x_pred2, y_pred2 = [], []
  x_p2 = original_x_2[i]
  y_p2 = original_y_2[i]
  x_pred1.append(x_p1)
  y_pred1.append(y_p1)
  x_pred2.append(x_p2)
  y_pred2.append(y_p2)
  p1_line.set_data(x_pred1, y_pred1)
  p2_line.set_data(x_pred2, y_pred2)

  x_af1, y_af1 = [], []
  x_af2, y_af2 = [], []
  if i >= T:
    x_a1 = Xped1[T:T+T]
    y_a1 = Yped1[T:T+T]
    x_a2 = Xped2[T:T+T]
    y_a2 = Yped2[T:T+T]
    x_af1 = x_af1 + x_a1
    y_af1 = y_af1 + y_a1
    x_af2 = x_af2 + x_a2
    y_af2 = y_af2 + y_a2
    af1_line.set_data(x_af1, y_af1)
    af2_line.set_data(x_af2, y_af2)
    
  x1_circle, y1_circle = [], []
  x2_circle, y2_circle = [], []
  if i >= T:
      for j in range(T):
          c_x1 = ConfPredRadius[j] * np.cos(ts) + Xped1[T+j]
          c_y1 = ConfPredRadius[j] * np.sin(ts) + Yped1[T+j]
          c_x2 = ConfPredRadius[j] * np.cos(ts) + Xped2[T+j]
          c_y2 = ConfPredRadius[j] * np.sin(ts) + Yped2[T+j]
          x1_circle.append(c_x1)
          y1_circle.append(c_y1)
          x2_circle.append(c_x2)
          y2_circle.append(c_y2)
          c1_line.set_data(x1_circle, y1_circle)
          c2_line.set_data(x2_circle, y2_circle)

  xh1_circle, yh1_circle = [], []
  xh2_circle, yh2_circle = [], []

  if i >= T:
      ch_x1 = ConfPredRadius[i-T] * np.cos(ts) + Xped1[i]
      ch_y1 = ConfPredRadius[i-T] * np.sin(ts) + Yped1[i]
      ch_x2 = ConfPredRadius[i-T] * np.cos(ts) + Xped2[i]
      ch_y2 = ConfPredRadius[i-T] * np.sin(ts) + Yped2[i]
      xh1_circle.append(ch_x1)
      xh2_circle.append(ch_x2)
      yh1_circle.append(ch_y1)
      yh2_circle.append(ch_y2)
      ch1_line.set_data(xh1_circle, yh1_circle)
      ch2_line.set_data(xh2_circle, yh2_circle)
  return p1_line, r_line, c1_line, af1_line, ch1_line, p2_line, c2_line, af2_line, ch2_line

import matplotlib
#matplotlib.use('Agg')
import matplotlib.animation
plt.legend(['Current_Obstacle_1', 'Robot', 'Safe_Area_1', 
            'Predicted_Obstacle_1', 'Current_Safe_Area_1','Current_Obstacle_2', 'Safe_Area_2', 
            'Predicted_Obstacle_2', 'Current_Safe_Area_2'], fontsize = 10)

plt.xlabel('Position x')
plt.ylabel('Position y')
plt.title("Open-loop")

for i in range(20):
    animate(i)
    plt.savefig("fig_%02i.png" % i)