import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as alg

# create the space and function
xList = np.linspace(-2.5, 2.5, 1000)
yList = np.linspace(-2.5, 2.5, 1000)
XP,YP = np.meshgrid(xList, yList)
ZP = XP**2 + 2 * YP**2

# create gradient and hessian of f
G = np.array([[2],[4]])
H = np.array([[2,0],[0,4]])

# set the staring position
X = np.array([[2.0],[2.0]])

# set the epsilon for stopping condition
eps = 1e-5

# create empty list to store points
x_list = [X[0]]
y_list = [X[1]]

iter = 0
while True:

    # increase the iteration number
    iter = iter + 1

    # compute the gradient vector
    GF = G * X

    # print some information
    print("Iter: ", iter, "Norm: ", alg.norm(GF))

    # if the stopping condition is satisfied, break the loop
    if alg.norm(GF) < eps:
        break

    # choose steepest descent direction as the negative gradient
    d = -GF

    # exact line search
    # method = 'exact_line_search'
    # eta1 = (d.T.dot(H.dot(X)))
    # eta2 = (d.T.dot(H.dot(d)))
    # eta = -(d.T.dot(H.dot(X))) / (d.T.dot(H.dot(d)))

    # small fixed step size
    method = 'small_fixed_step_size'
    eta = 0.05

    # do the computataion
    X = X + eta * d

    # add the results to the list
    x_list.append(X[0])
    y_list.append(X[1])

# create the plots
fig = plt.figure()
ax = fig.add_subplot(111)

# illustrate the function using contours
plt.contour(XP, YP, ZP, 30, alpha=0.7, zorder=1)
plt.xlabel('x')
plt.ylabel('y')
ax.set_aspect('equal')

# plot the steps
plt.plot(x_list, y_list, ls='-', color='royalblue', marker='o', mfc='orange', mec='orange', linewidth=3, markersize=5, zorder=2)
plt.scatter(x_list[0], y_list[0], c='black', s=30, zorder=3)

# save the result
filename = method + '_eta_' + str(eta) + '_iter_' + str(iter) + '.png'
plt.savefig(filename, bbox_inches='tight', dpi=300)

# display the results
plt.show()


