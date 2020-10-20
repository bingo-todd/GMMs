import numpy as np
from matplotlib import ticker, cm
import matplotlib.pyplot as plt


def norm_01(x):
    """Normlize the range of each variable in x to [0,1], pre-processing of
    k_means
    Args: x, [n_sample,n_var]
    """
    x_min = np.min(x,axis=0)
    x_max = np.max(x,axis=0)
    x_norm = np.divide(x-x_min,x_max-x_min)
    return [x_norm,[x_min,x_max]]


def inv_norm_01(x_norm,norm_params):
    x_min,x_max = norm_params
    x = np.multiply(x_norm,x_max-x_min)+x_min
    return x


def k_means(x,k,is_plot=False,max_iter=1000,epsilon=1e-20):
    """used to determine the initial mean of GM
    Args:
        x: data, [n_sample,n_variable]
        k: the number of clusters
        max_iter: the maximal iterations allowed
        epsilon: the threshold of difference between two successive iterations
    Returns:
        centers of clusters
    """
    x_norm,norm_params = norm_01(x)
    n_sample = x_norm.shape[0]

    dist_matrix = np.zeros((n_sample,k))
    centers_init = x_norm[np.random.choice(n_sample,size=k,replace=False),:]
    centers = centers_init.copy()
    n_iter = 0
    while True:
        centers_old = centers.copy()
        n_iter = n_iter+1
        for i in range(k):
            dist_matrix[:,i]= 1/(np.sum((x_norm-centers[i])**2,axis=1)+1e-20)
        belongship = (dist_matrix.T/np.sum(dist_matrix,axis=1).T).T
        for i in range(k):
            centers[i,:] = np.dot(x_norm.T,belongship[:,i])/np.sum(belongship[:,i])
        # centers_list.append(centers.copy())

        # stop criteria
        if n_iter>max_iter:
            break
        if np.sum((centers - centers_old)**2) < epsilon:
            break

    centers_init = inv_norm_01(centers_init,norm_params)
    centers = inv_norm_01(centers,norm_params)

    if is_plot:
        # if variables number exceed 2, PCA may be used
        fig,ax = plt.subplots(1,2)
        ax[0].plot(x[:,0],x[:,1],'x',alpha=0.5)
        ax[0].plot(centers_init[:,0],centers_init[:,1],'ro')
        ax[0].set_title('before')
        ax[0].set_aspect('equal','box')

        labels = np.argmax(belongship,axis=1)
        for cluster_i in range(k):
            X_in_cluster_i = x[labels==cluster_i,:]
            lines = ax[1].plot(X_in_cluster_i[:,0],X_in_cluster_i[:,1],'x',
                               label='cluster {}'.format(cluster_i),alpha=0.5)
            ax[1].plot(centers[cluster_i,0],centers[cluster_i,1],'o',
                       color=lines[0].get_color(),markersize=10)
        ax[1].legend()
        ax[1].set_title('after')
        ax[1].set_aspect('equal','box')
        plt.tight_layout()
        return [centers,fig]
    else:
        return centers
