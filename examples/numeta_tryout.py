import numpy as np
import numeta as nm


@nm.jit
def saxpy(alpha, x, y):
    for i in nm.range(x.shape[0]):
        y[i] = alpha * x[i] + y[i]


@nm.jit
def moving_average(radius: nm.comptime, x, y):
    n = x.shape[0]
    for i in nm.range(n):
        acc = nm.scalar(nm.f8)
        acc[:] = 0.0
        for k in range(-radius, radius + 1):
            j = i + k
            with nm.If((j >= 0) & (j < n)):
                acc[:] += x[j]
        y[i] = acc / (2 * radius + 1)


@nm.jit
def signed_sqrt(x, out):
    for i in nm.range(x.shape[0]):
        with nm.If((i / 2) * 2 == i):
            out[i] = nm.sqrt(x[i])
        with nm.Else():
            out[i] = -nm.sqrt(x[i])


def main():
    x = np.arange(8, dtype=np.float64)
    y = np.ones_like(x)
    saxpy(2.0, x, y)
    print("saxpy:", y)

    src = np.linspace(0.0, 7.0, 8, dtype=np.float64)
    avg = np.zeros_like(src)
    moving_average(1, src, avg)
    print("moving_average(radius=1):", avg)

    values = np.arange(1, 9, dtype=np.float64)
    signed = np.zeros_like(values)
    signed_sqrt(values, signed)
    print("signed_sqrt:", signed)


if __name__ == "__main__":
    main()
