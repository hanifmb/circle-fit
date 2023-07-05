import numpy as np
from scipy.optimize import least_squares

def circle_residuals(params, x, y):
    """
    Residual function for circle fitting.
    params: [a, b, r] - circle parameters (center coordinates: a, b and radius: r)
    x, y: Data points coordinates
    """
    a, b, r = params
    return (x - a) ** 2 + (y - b) ** 2 - r ** 2

def fit_circle(x, y):
    """
    Fit a circle to a set of points using the least squares method.
    x, y: Data points coordinates
    Returns the circle parameters: (a, b, r)
    """
    # Initial guess for the circle parameters (a, b, r)
    x_m = np.mean(x)
    y_m = np.mean(y)
    r_guess = np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2).mean()
    params_guess = [x_m, y_m, r_guess]

    # Perform least squares optimization
    result = least_squares(circle_residuals, params_guess, args=(x, y))

    # Extract the optimized circle parameters
    a_opt, b_opt, r_opt = result.x
    return a_opt, b_opt, r_opt

def main():
    # Example usage
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])

    a, b, r = fit_circle(x, y)
    print(f"Center: ({a}, {b}), Radius: {r}")

if __name__ == "__main__":
    main()

