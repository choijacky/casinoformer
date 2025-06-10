import math
import numpy as np

from np_attention import get_proj, kernel
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="FAVOR+", help='one of "FAVOR+", "FAVOR+ReLU", "positive_alignment", "RandomFourierFeatures"')
    parser.add_argument('--D', type=int, default=64, help="D")
    parser.add_argument('--R', type=int, default=128, help='R')
    parser.add_argument('--orth', action="store_true", help='set to False if R > 2*D')
    parser.add_argument('--query_norm_value', type=float, default=8, help="norm of query")
    parser.add_argument('--key_norm_value', type=float, default=8, help="norm of key")
    parser.add_argument('--scaling', type=float, default=1, help="additional scaling factor applied to inner product (or other quantities) to adjust variance/discriminative behaviour")
    parser.add_argument('--exp', type=float, default=1.0, help='this is used for FAVOR+ReLU. A high exponent makes it more discriminative at the cost of higher variance. Theoretical mean only known for kernel_exponent = 1.')

    args = parser.parse_args()

    # settings should match what is used in GPT
    method = args.method # one of "FAVOR+", "FAVOR+ReLU", "positive_alignment", "RandomFourierFeatures"
    D = args.D
    assert args.R % 2 == 0
    R = args.R if method != "RandomFourierFeatures" else args.R // 2
    orthogonal_features = args.orth # set to False if R > D and method = "RandomFourierFeatures" or if R > 2*D in case of the other methods
    key_norm = args.key_norm_value # sqrt(D), but can try out other norms. In FAVOR+ a high norm is more discriminative, but has higher variance
    query_norm = args.query_norm_value
    scaling = args.scaling # additional scaling factor applied to inner product (or other quantities) to adjust variance/discriminative behaviour

    kernel_exponent = args.exp # this is used for FAVOR+ReLU. A high exponent makes it more discriminative at the cost of higher variance. Theoretical mean only known for kernel_exponent = 1.

    x_axis_resolution = 200
    samples_for_expectation = 500

    # avoids zero std. => leads to weird results on plot
    epsilon = 10**-10

    # generates two orthogonal vectors of unit norm. This allows to interpolate from rho = -1 to rho = 1.
    U, _ = np.linalg.qr(np.random.randn(D, 2), mode='reduced')
    k = U[:, 0]
    k_orth = U[:, 1]

    # rescales orthogonal vectors to desired norm
    k = k * key_norm
    k_orth = k_orth * key_norm

    # quantities will be plotted with rho on x-axis
    rho = []
    mean_dot_products = []
    std_dot_products = []
    rel_std_dot_products = []

    # iterate over rho and record quantities
    for i in range(x_axis_resolution + 1):
        # query rotates from totally anti-aligned to totally aligned
        q = ( np.cos(math.pi + i / x_axis_resolution * math.pi) * k\
            + np.sin(math.pi + i / x_axis_resolution * math.pi) * k_orth) * query_norm / key_norm
        # add new entry to quantities which will be plotted
        rho.append(2 * i / x_axis_resolution - 1)
        mean_dot_products.append(0)
        std_dot_products.append(0)
        rel_std_dot_products.append(0)

        # evaluate expectation for mean and std.
        for s in range(samples_for_expectation):
            projections = get_proj(R=R, D=D, orthogonal=orthogonal_features, hyperbolic=(method != "RandomFourierFeatures"), constant_norm=True)#(method == "FAVOR+ReLU"))
            dot = np.dot(kernel(q.reshape((1, 1, D)), projections, method, kernel_exponent, scaling).reshape((R if method != "RandomFourierFeatures" else 2*R,)), # scaling for 1/sqrt(D)
                        kernel(k.reshape((1, 1, D)), projections, method, kernel_exponent, scaling).reshape((R if method != "RandomFourierFeatures" else 2*R,))) # scaling for 1/sqrt(D)
            dot = (dot > 0) * dot # for RandomFourierFeatures, get rid of negative values
            mean_dot_products[i] += dot
            std_dot_products[i] += dot ** 2

        mean_dot_products[i] /= samples_for_expectation
        # unbiased estimator of std: sqrt((sum_i x_i^2 - n *(sum_i x_i / n)^2)/(n-1))
        std_dot_products[i] -= samples_for_expectation * mean_dot_products[i]**2
        std_dot_products[i] /= (samples_for_expectation-1)
        std_dot_products[i] = (std_dot_products[i] > 0) * std_dot_products[i] # numerical stability
        std_dot_products[i] = np.sqrt(std_dot_products[i]) +  epsilon# numerical stability
        rel_std_dot_products[i] = std_dot_products[i] / mean_dot_products[i]

    identifier = f'method: {method}, D: {D}, R: {R}, orthogonal features: {orthogonal_features}, key norm: {key_norm}, query norm: {query_norm}, kernel exponent: {kernel_exponent}, samples for expectation: {samples_for_expectation}, x-axis resolution: {x_axis_resolution}'
    print("")
    print(identifier)

    print(f"mean at {rho[0]}: {mean_dot_products[0]}")
    print(f"std. at {rho[0]}: {std_dot_products[0]}")

    print(f"mean at {rho[x_axis_resolution//2]}: {mean_dot_products[x_axis_resolution//2]}")
    print(f"std. at {rho[x_axis_resolution//2]}: {std_dot_products[x_axis_resolution//2]}")

    print(f"mean at {rho[x_axis_resolution]}: {mean_dot_products[x_axis_resolution]}")
    print(f"std. at {rho[x_axis_resolution]}: {std_dot_products[x_axis_resolution]}")


    fig, axs = plt.subplots(4)
    fig.suptitle(identifier)
    axs[0].plot(rho, mean_dot_products)
    axs[0].set_xlabel("rho")
    axs[0].set_ylabel("mean linear scale")
    axs[1].semilogy(rho, mean_dot_products)
    axs[1].set_xlabel("rho")
    axs[1].set_ylabel("mean log scale")
    axs[2].plot(rho, rel_std_dot_products)
    axs[2].set_xlabel("rho")
    axs[2].set_ylabel("std/mean linear scale")
    axs[2].axhline(y=1,xmin=-1,xmax=1,c="blue",linewidth=0.5)
    axs[3].semilogy(rho, rel_std_dot_products)
    axs[3].set_xlabel("rho")
    axs[3].set_ylabel("std/mean log-scale")
    axs[3].axhline(y=1,xmin=-1,xmax=1,c="blue",linewidth=0.5)

    plt.show()