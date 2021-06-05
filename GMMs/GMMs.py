import gif
import pickle
import numpy as np
import matplotlib.pyplot as plt
from BasicTools import plot_tools
from .k_means import k_means


class GMMs(object):
    def __init__(self, x=None, k=None, lh_theta=1e-10, max_iter=2000):
        """
        Args:
            x: data to be modeled
            k: component number of GMMs
            lh_theta: threshold of Euclidean distance mius of two successive
                     iterations
            max_iter: maximum iteration allowed
        """
        self.x = x
        self.k = k
        self.lh_theta = lh_theta
        self.max_iter = max_iter

        #
        self.gmms_params = [None, None, None]  # mu_all, sigma_all, pi_all
        self.train_record = {'gmms_params': [], 'lh': []}
        self.EPS = 1e-10  # small value to avoid #overflow

    def _gm_pdf(self, x, mu, sigma):
        """ the probability density of given data
        Args:
            x: input data, [n_sample,n_var]
            mu: mean of Gaussian model
            sigma: covariance matrices of Gaussian model
        Returns:
            probability density of x, [n_sample,n_variable]
        """
        n_var = mu.shape[-1]
        inv_sigma = np.linalg.inv(sigma)

        if np.isnan(1/np.linalg.det(sigma)):
            print('det', np.linalg.det(sigma), 'sigma', sigma)
            print('det of sigma is too small')
            return np.zeros(0)

        diff = x-mu
        dist = np.sum(
                np.multiply(diff,
                            np.dot(inv_sigma, diff.T).T),
                axis=1)
        p = ((np.linalg.det(sigma))**(-.5)
             * ((2*np.pi)**(-.5*n_var))
             * np.exp(-.5*dist))
        return p

    def _gmm_pdf(self, x, mu_all, sigma_all, pi_all, return_raw=False):
        """the probability density of given data
        Args:
            x: input data, [n_sample,n_var]
            mu_all: mean of Gaussian models
            sigma_all: covariance matrices of Gaussian models
            pi_all: weights of each Gaussian model
        Returns:
            probability density of x, [n_sample,n_variable]
        """
        n_sample = x.shape[0]
        k = mu_all.shape[0]
        p = np.zeros((n_sample, k))
        for model_i in range(k):
            p[:, model_i] = self._gm_pdf(
                x, mu_all[model_i, :], sigma_all[model_i, :, :])
        if not return_raw:
            p = np.sum(p*pi_all[np.newaxis, :], axis=1)
        return p

    def _init_gmms_params(self, init_method):
        """"""
        x = self.x
        k = self.k
        [n_sample, n_var] = x.shape
        # mu
        if init_method == 'k_means':
            mu_all = k_means(x=x, k=k)
        elif init_method == 'random':
            mu_all = x[np.random.choice(n_sample, size=k, replace=False), :]
        # sigma
        sigma_all = np.zeros((k, n_var, n_var), dtype=np.float32)
        for model_i in range(k):
            sigma_all[model_i] =\
                np.diag(np.var(self.x-mu_all[model_i], axis=0))
        # pi
        pi_all = np.ones(k, dtype=np.float32)*1.0/k

        gmms_params = [mu_all, sigma_all, pi_all]
        return gmms_params

    def _is_stop(self):
        """ return true if likelihood change between two successive iteration
         do not exceed self.lh_theta
        """
        if (len(self.train_record['lh']) > 1 and
            np.abs(self.train_record['lh'][-1]
                   - self.train_record['lh'][-2]) < self.lh_theta):
            return True
        else:
            return False

    def EM(self, init_method='random', constrain_sigma_diag=True,
           is_plot=False, fig_dir=None,
           is_log=False):
        """Expecation-Maximization algorithm
        Args:
            init_method: method of initialing the center of Gaussian models,
                'random'
                'k_means'
                'norm_#': equally distributed along give axis
            n_iter_max: the maximum of iteration
            is_plot: whether plot the training result
            fig_path: file path where image to be saved
        """

        k = self.k
        x = self.x
        [n_sample, n_var] = x.shape

        [mu_all, sigma_all, pi_all] = self._init_gmms_params(init_method)
        lh = np.mean(
                np.log(
                    self._gmm_pdf(x, mu_all, sigma_all, pi_all)
                    + self.EPS))

        #
        self.train_record['gmms_params'] =\
            [[mu_all.copy(), sigma_all.copy(), pi_all.copy()]]
        self.train_record['lh'] = [lh]

        # condition probability of data, intialized as 0
        posterior = np.zeros((n_sample, k), dtype=np.float32)
        n_iter = 0
        is_except = False
        except_info = None
        for iter in range(1, self.max_iter+1):
            try:
                # E step
                for model_i in range(k):
                    posterior[:, model_i] = self._gm_pdf(x, mu_all[model_i],
                                                         sigma_all[model_i])
                # M step
                posterior = np.divide(
                                posterior,
                                (np.sum(posterior, axis=1)[:, np.newaxis]
                                 + self.EPS))
                for model_i in range(k):
                    if np.isnan(np.max(posterior[:, model_i])):
                        raise ValueError('probability density of data is nan')

                    pd_data_model_i = np.sum(posterior[:, model_i])
                    pi_all[model_i] = pd_data_model_i/n_sample
                    mu_all[model_i, :] = (np.dot(x.T, posterior[:, model_i])
                                          / pd_data_model_i)
                    diff_array = x - mu_all[model_i, :]
                    sigma_all[model_i] =\
                        (np.dot((diff_array.T*posterior[:, model_i]),
                                diff_array)
                         / pd_data_model_i)
                    if constrain_sigma_diag:
                        sigma_max = np.max(sigma_all[model_i])
                        for var_i in range(sigma_all[model_i].shape[0]):
                            sigma_all[model_i][var_i, var_i] =\
                                np.clip(sigma_all[model_i][var_i, var_i],
                                        sigma_max*self.EPS,
                                        np.Inf)
                            sigma_all[model_i]+np.eye(n_var)*self.EPS

                lh = np.mean(
                        np.log(self._gmm_pdf(x, mu_all, sigma_all, pi_all)
                               + self.EPS))

                self.train_record['gmms_params'].append(
                    [mu_all.copy(), sigma_all.copy(), pi_all.copy()])
                self.train_record['lh'].append(lh)

                n_iter = n_iter+1
                if self._is_stop():
                    break

                if is_log:
                    print('iter {} lh {}'.format(iter, lh))
                if is_plot:
                    fig, ax = self.visualize(
                        gmms_params=[mu_all, sigma_all, pi_all])
                    fig.savefig(f'{fig_dir}/step-{iter}.png')
                    plt.close()

            except Exception as e:
                except_info = e
                is_except = True
                n_iter = n_iter-1
                break

        self.gmms_params = [mu_all, sigma_all, pi_all]

        if is_except:
            raise Exception(except_info)

    def plot_train_gif(self, gif_path, duration=3.5):

        @gif.frame
        def plot_gif_frame(gmms_params):
            self.visualize(gmms_params=gmms_params)

        frames = []
        for gmms_params in self.train_record['gmms_params']:
            frame = plot_gif_frame(gmms_params=gmms_params)
            frames.append(frame)
        gif.save(
            frames, gif_path, duration=duration, unit='s', between='startend')

    def plot_train_curve(self, fig_path):
        x = self.x
        figsize = plot_tools.get_figsize(1, 3)
        fig, ax = plt.subplots(1, 3, figsize=figsize)

        # Befor EM updating
        self.visualize(
            ax=ax[0], x=x, gmms_params=self.train_record['gmms_params'][0])
        ax[0].set_title('Before EM')

        # After EM updating
        self.visualize(
            ax=ax[1], x=x, gmms_params=self.train_record['gmms_params'][-1])
        ax[1].set_title('After EM')

        ax[2].plot(self.train_record['lh'])
        bic = self.cal_bic()
        ax[2].text(
            0.6, 0.2, f'bic: {bic:.2f}', transform=ax[2].transAxes,
            multialignment='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if fig_path is not None:
            fig.savefig(fig_path)
        plt.close(fig)

    def visualize(self, x=None, gmms_params=None, ax=None, fig=None):
        """ GMMs(contours) as well as data(scatter)
        Args:
            x: numpy ndarray with shape of [n_sample, n_var]. if not given,
                data for GMMs settings will be used
            gmms_prams: [mu_all, sigma_all, pi_all]. if not given,
                current GMMs params will be used
        Returns:
            figure
        """
        if x is None:
            x = self.x

        if gmms_params is None:
            gmms_params = self.gmms_params
        mu_all, sigma_all, pi_all = gmms_params

        if ax is None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)

        plot_tools.plot_line(ax, x[:, 0], x[:, 1], 'bo', alpha=0.3, zorder=1,
                             markersize=3)
        # make grid within specified region
        n_point = 1000
        x0, x1 = np.meshgrid(
            np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), n_point),
            np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), n_point))
        x0x1 = np.array([x0.flatten(), x1.flatten()]).T
        # cal pdf of each point on grid
        p = self._gmm_pdf(x0x1, mu_all, sigma_all, pi_all)
        # divide p range in to 100 segments
        levels = np.linspace(np.min(p), np.max(p), 100)
        # assigment each data to corresponding segments
        try:
            index_round = np.argmin(
                np.abs(p[:, np.newaxis]-levels[np.newaxis, :]), axis=1)
        except Exception as e:
            fig, ax = plt.subplots(1, 1)
            ax.plot(p)
            fig.savefig('gmms_p.png')
            raise Exception(e)

        for i in range(100):
            p[index_round == i] = levels[i]

        p = p.reshape([x0.shape[0], x1.shape[0]])
        plot_tools.plot_contour(
            ax, x0, x1, p,
            levels=[levels[i] for i in range(0, 100, 20)],
            alpha=1, zorder=2, colors='k')
        plot_tools.plot_line(ax, mu_all[:, 0], mu_all[:, 1], 'ro',
                             markersize=3, zorder=3)
        return fig, ax

    def save(self, gmms_params_path, train_record_path=None):
        with open(gmms_params_path, 'wb') as gmms_params_f:
            pickle.dump(self.gmms_params, gmms_params_f)

        if train_record_path is not None:
            with open(train_record_path, 'wb') as train_record_f:
                pickle.dump(self.train_record, train_record_f)

    def load(self, gmms_params_path, train_record_path=None):
        with open(gmms_params_path, 'rb') as gmms_params_f:
            self.gmms_params = pickle.load(gmms_params_f)

        if train_record_path is not None:
            with open(train_record_path, 'rb') as train_record_f:
                self.train_record = pickle.load(train_record_f)

    def cal_lh(self, x):
        mu_all, sigma_all, pi_all = self.gmms_params
        return self._gmm_pdf(x, mu_all, sigma_all, pi_all)

    def cal_bic(self, x=None):
        """ Bayesian information criterion
        """
        if x is None:
            x = self.x

        n_sample, n_var = self.x.shape
        n_params = self.k*n_var + self.k*n_var**2 + self.k  # mean+sigma+pi
        bic = (n_params*np.log(n_sample)
               - 2*np.sum(np.log(self.cal_lh(x)+self.EPS)))
        return bic


class GMMs_classifier:
    def __init__(self):
        self.model_all = []
        self.label_all = []

    def load(self, model_path_all, label_all):
        self.model_all.clear()
        self.label_all.clear()
        for model_path, label in zip(model_path_all, label_all):
            gmms = GMMs()
            gmms.load(model_path=model_path)
            self.model_all.append(gmms)
            self.label_all.append(label)

    def classify(self, x):
        prob_all = np.asarray([model.cal_lh(x) for model in self.model_all])
        label_est = np.asarray([self.label_all[i]
                                for i in np.argmax(prob_all, axis=0)])
        return label_est
