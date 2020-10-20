import numpy as np
from BasicTools import plot_tools
from sklearn.datasets import make_blobs
from GMMs import GMMs, k_means


def k_means_test():
    x, Y = make_blobs(cluster_std=1, random_state=np.random.randint(100),
                      n_samples=500, centers=3)
    # Stratch dataset to get ellipsoid data
    # x = np.dot(x,np.random.RandomState(0).randn(2,2))

    centers, fig = k_means(x, k=3, is_plot=True)
    plot_tools.savefig(fig)


def GMMs_test():
    X, Y = make_blobs(cluster_std=1, random_state=np.random.randint(100),
                      n_samples=800, centers=3)

    gmms = GMMs(x=X, k=3, max_iter=100, lh_theta=1e-5)
    gmms.EM(init_method='random',
            is_plot=True, fig_fpath='images/example.png',
            is_gif=True, gif_fpath='images/example.gif')


if __name__ == '__main__':
    # k_means_test()
    GMMs_test()
