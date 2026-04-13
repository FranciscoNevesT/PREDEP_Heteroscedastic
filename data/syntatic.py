import numpy as np


def step_wise(num_points,f, start_std=0.1, end_std=0.5, mode = 'normal'):
    x = np.random.rand(num_points)
    y = f(x)
    y = (y - np.mean(y)) / np.std(y)  # Normalize y to have mean 0 and std 1

    if mode == 'normal':
        x_mode = x
    elif mode == 'reflect':
        x_mode = np.abs(x - 0.5) * 2  # Reflect x around 0.5 to create a step-wise pattern
    elif mode == 'cosine':
        x_mode = (1 - np.cos(16 * np.pi * x))/2  # Cosine-based pattern
    else:
        raise ValueError("Invalid mode. Choose from 'normal', 'reflect', or 'cosine'.")

    std_vector = x_mode * (end_std - start_std) + start_std  # Linear interpolation of std based on x

    y = y + np.random.normal(0, std_vector)
    return x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def f(x):
        return x

    x, y = step_wise(10000, f, start_std=0.1, end_std=1.0)

    plt.scatter(x, y, s=10, alpha=0.5)
    plt.title("Step-wise Noise")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

