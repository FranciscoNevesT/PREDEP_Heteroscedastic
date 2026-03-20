import numpy as np


def step_wise(num_points,f,num_steps, start_std=0.1, end_std=0.5):
    x = np.random.rand(num_points)
    y = f(x)

    step_size = 1 / num_steps
    step_edges = np.arange(0, 1 + step_size, step_size)
    
    std_vector = np.zeros(num_points)
    for step in range(num_steps):
        points = (x >= step_edges[step]) & (x < step_edges[step + 1])
        std = start_std + (end_std - start_std) * (step / (num_steps - 1))  # Linear interpolation of std
        std_vector[points] = std

    #y = y + np.random.normal(0, 1, size=y.shape) 

    y = y + np.random.normal(0, std_vector)

    #print(f"np.std(y): {np.std(y)}, np.mean(y): {np.mean(y)}")
    #print(f"np.std(y - f(x)): {np.std(y - f(x))}, np.mean(y - f(x)): {np.mean(y - f(x))}")
    return x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def f(x):
        return x

    x, y = step_wise(10000, f, num_steps=1, start_std=0.1, end_std=1.0)

    plt.scatter(x, y, s=10, alpha=0.5)
    plt.title("Step-wise Noise")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

