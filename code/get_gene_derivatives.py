import numpy as np
import matplotlib.pyplot as plt


def get_derivatives(x,y, visualize = False):
    from scipy.interpolate import make_splrep, splev
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Fit spline representation (larger s is smoother)
    tck = make_splrep(x_sorted, y_sorted, s=len(x_sorted) * np.var(y_sorted) / 1)
#     x_fit = np.linspace(x_sorted.min(), x_sorted.max(), 500)
    x_fit = np.linspace(0, 1, 500)
    y_fit = splev(x_fit, tck)                 # function value
    dy_dx = splev(x_fit, tck, der=1)          # first derivative
    d2y_dx2 = splev(x_fit, tck, der=2)        # second derivative

    if visualize:
        plt.figure(figsize=(4, 5))
        plt.scatter(x, y, s=1, alpha=0.3, label='raw data')
        plt.plot(x_fit, y_fit, color='black', label='spline fit')
        plt.plot(x_fit, dy_dx, color='blue', label='1st derivative')
#         plt.plot(x_fit, d2y_dx2, color='red', label='2nd derivative')
        plt.legend()
        plt.show()
    return(dy_dx, d2y_dx2,x_fit, y_fit)