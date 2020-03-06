import numpy as np
from GMMs import GMMs
from sklearn.datasets import make_blobs


if __name__ == '__main__':
    X,Y = make_blobs(cluster_std=1, random_state=np.random.randint(100),
                     n_samples=800, centers=3)

    gmms = GMMs(x=X, k=3, max_iter=100, lh_theta=1e-5)
    gmms.EM(init_method='random',
            is_plot=True, fig_fpath='images/example.png',
            is_gif=True, gif_fpath='images/example.gif')
