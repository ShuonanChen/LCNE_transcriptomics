import numpy as np
import matplotlib.pyplot as plt


def get_derivatives(x,y, visualize = False):
    from scipy.interpolate import make_splrep, splev,splrep
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    interior_knots = np.linspace(x_sorted.min(),x_sorted.max(),5)[1:-1]
    tck = splrep(x_sorted, y_sorted,k=3,t=interior_knots)
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = splev(x_fit, tck)                 # function value
    dy_dx = splev(x_fit, tck, der=1)          # first derivative
    d2y_dx2 = splev(x_fit, tck, der=2)        # second derivative
    check_idx = np.where((x_fit>0) &(x_fit<1))[0]
    
    y_pred = splev(x_sorted, tck)
    resid = y_sorted - y_pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_sorted - np.mean(y_sorted))**2)
    r2 = 1 - ss_res / ss_tot

    
    
    if visualize:
        plt.figure(figsize=(4, 5))
        plt.scatter(x, y, s=1, alpha=0.3, label='raw data')
        plt.plot(x_fit, y_fit, color='black', label='spline fit')
        plt.plot(x_fit, dy_dx, color='blue', label='1st derivative')
        plt.legend()
        plt.show()
    return(dy_dx[check_idx], d2y_dx2[check_idx],x_fit[check_idx], y_fit[check_idx],r2)
#     return(dy_dx, d2y_dx2,x_fit, y_fit)

