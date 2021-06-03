import os
import pickle
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from BasicTools import plot_tools
from GMMs import GMMs, k_means


def k_means_test():
    x, Y = make_blobs(cluster_std=1, random_state=np.random.randint(100),
                      n_samples=500, centers=3)
    # Stratch dataset to get ellipsoid data
    # x = np.dot(x,np.random.RandomState(0).randn(2,2))

    centers, fig = k_means(x, k=3, is_plot=True)
    plot_tools.savefig(fig)


def gen_random_dataset(n_center, n_sample):
    dataset_path = f'dataset-center_{n_center}-sample_{n_sample}.pkl'
    random_state = np.random.RandomState()
    if not os.path.exists(dataset_path):
        X, Y = make_blobs(
            cluster_std=random_state.rand()*10,
            random_state=random_state,
            n_samples=n_sample, centers=3)
        with open(dataset_path, 'wb') as dataset_f:
            pickle.dump([X, Y], dataset_f)
    else:
        with open(dataset_path, 'rb') as dataset_f:
            X, Y = pickle.load(dataset_f)
    return X, Y


def GMMs_test():

    X, Y = gen_random_dataset(4, 4000)

    k_all = np.arange(2, 12, 2)
    n_k = k_all.shape[0]
    pic_record = np.arange(n_k)
    for i, k in enumerate(k_all):
        gmms = GMMs(x=X, k=k, max_iter=2000, lh_theta=1e-10)
        gmms.EM(init_method='random')
        bic = gmms.cal_bic()
        pic_record[i] = bic
        gmms.plot_train_curve(f'train_record-{k}.png')
        fig, ax = gmms.visualize()
        ax.set_title(f'{k}: {bic:.2f}')
        fig.savefig(f'visualize-{k}.png')

    fig, ax = plt.subplots(1, 1)
    ax.plot(k_all, pic_record)
    ax.set_xlabel('k')
    ax.set_ylabel('PIC')
    fig.savefig('pic_k_func.png')


if __name__ == '__main__':
    # k_means_test()
    GMMs_test()
