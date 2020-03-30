import numpy as np
import matplotlib.pyplot as plt
from BasicTools import plot_tools
from .k_means import k_means


class GMMs(object):
    def __init__(self,x=None,k=None,lh_theta=1e-10,max_iter=2000):
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

        self.iter_record = {'mu_all':[],
                            'sigma_all':[],
                            'pi_all':[],
                            'lh':[]}


    def _gm_pdf(self,x,mu,sigma):
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
            print('det', np.linalg.det(sigma),'sigma',sigma)
            print('det of sigma is too small')
            return np.zeros(0)

        diff = x-mu
        dist = np.sum(np.multiply(diff,np.dot(inv_sigma,diff.T).T),axis=1)
        p = (np.linalg.det(sigma))**(-.5) * ((2*np.pi)**(-.5*n_var))*np.exp(-.5*dist)
        return p


    def _gmm_pdf(self,x,mu_all,sigma_all,pi_all):
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
        p = 0
        for model_i in range(k):
            p = p + pi_all[model_i]*self._gm_pdf(x,mu_all[model_i,:],
                                                   sigma_all[model_i,:,:])
        return p


    def _init_params(self,init_method):
        """"""
        x = self.x
        k = self.k
        [n_sample,n_var] = x.shape
        # mu
        if init_method == 'k_mean':
            mu_all=k_means.k_means(x=x,k=k)
        elif len(init_method)>5 and init_method[:5] == 'norm_':
            mu_all = np.repeat(np.mean(x,0)[np.newaxis,:], k, axis=0)
            axis = np.int16(init_method[5:])
            max_ax, min_ax = [np.max(x, 0)[axis], np.min(x, 0)[axis]]
            mu_all[:, axis] = np.linspace(max_ax, min_ax, k)
        elif init_method == 'random':
            mu_all = x[np.random.choice(n_sample,size=k,replace=False),:]

        # sigma
        sigma_all = np.zeros((k,n_var,n_var),dtype=np.float32)
        mean_var = np.mean(np.var(x,axis=0))
        for model_i in range(k):
            sigma_all[model_i] = np.eye(n_var)*mean_var/(10*n_var)

        # pi
        pi_all = np.ones(k,dtype=np.float32)*1.0/k

        return [mu_all,sigma_all,pi_all]


    def _update_record(self,mu_all,sigma_all,pi_all,lh):
        self.iter_record['mu_all'].append(mu_all.copy())
        self.iter_record['sigma_all'].append(sigma_all.copy())
        self.iter_record['pi_all'].append(pi_all.copy())
        self.iter_record['lh'].append(lh)


    def _clear_record(self):
        self.iter_record['mu_all'].clear()
        self.iter_record['sigma_all'].clear()
        self.iter_record['pi_all'].clear()
        self.iter_record['lh'].clear()


    def _is_stop(self):
        """ return true if likelihood change between two successive iteration
         do not exceed self.lh_theta
        """
        if (len(self.iter_record['lh'])>1 and
            np.abs(self.iter_record['lh'][-1]
                   -self.iter_record['lh'][-2]) < self.lh_theta):
            return True
        else:
            return False


    def EM(self, init_method='random', is_log=False, is_plot=False,
           fig_fpath=None, is_gif=False, gif_fpath=None, fps=10):
        """Expecation-Maximization algorithm
        Args:
            init_method: method of initialing the center of Gaussian models,
                'random'
                'k_means'
                'norm_#': equally distributed along give axis
            n_iter_max: the maximum of iteration
            is_plot: whether plot the training result
            fig_fpath: file path where image to be saved
            is_gif: whether to plot the iteration process in gif
            gig_fpath: file path where gif to be saved
        """
        epsilon = 1e-20 # small value to avoid #overflow

        k = self.k
        x = self.x
        [n_sample,n_var] = x.shape

        [mu_all,sigma_all,pi_all] = self._init_params(init_method)
        lh = np.mean(np.log(self._gmm_pdf(x,mu_all,sigma_all,pi_all)+epsilon))
        self._clear_record()
        self._update_record(mu_all,sigma_all,pi_all,lh)

        # condition probability of data, intialized as 0
        posterior = np.zeros((n_sample,k),dtype=np.float32)
        n_iter = 0
        is_except = False
        except_info = None
        for iter in range(1,self.max_iter+1):
            try:
                # E step
                for model_i in range(k):
                    posterior[:,model_i] = self._gm_pdf(x,mu_all[model_i],
                                                      sigma_all[model_i])

                # M step
                posterior = np.divide(posterior,
                                np.sum(posterior,axis=1)[:,np.newaxis]+epsilon)
                for model_i in range(k):
                    if np.isnan(np.max(posterior[:,model_i])):
                        raise ValueError('probability density of data is nan')

                    pd_data_model_i = np.sum(posterior[:,model_i])
                    pi_all[model_i] = pd_data_model_i/n_sample
                    mu_all[model_i,:] = (np.dot(x.T,posterior[:,model_i])
                                         /pd_data_model_i)
                    diff_array = x - mu_all[model_i,:]
                    sigma_all[model_i,:,:] = ((np.dot(np.multiply(diff_array.T,
                                                        posterior[:,model_i]),
                                                     diff_array)
                                              /pd_data_model_i)
                                              +np.eye(n_var)*epsilon)

                lh = np.mean(np.log(self._gmm_pdf(x,mu_all,sigma_all,pi_all)
                                    +epsilon))
                self._update_record(mu_all,sigma_all,pi_all,lh)

                n_iter = n_iter+1
                if self._is_stop():
                    break

                if is_log:
                    print('iter {} lh {}'.format(iter,lh))

            except Exception as e:
                except_info = e
                is_except = True
                n_iter = n_iter-1
                break

        self.mu_all = mu_all
        self.sigma_all = sigma_all
        self.pi_all = pi_all

        if is_gif:
            gif_writer = plot_tools.Gif()
            fig,ax = plt.subplots(1,1)
            for iter in range(n_iter+1):
                gif_writer.add(self.plot(ax,x,
                               self.iter_record['mu_all'][iter],
                               self.iter_record['sigma_all'][iter],
                               self.iter_record['pi_all'][iter]))
            gif_writer.save(gif_fpath,fig,fps=fps)

        if is_plot:
            self.plot_record(fig_fpath)

        if is_except:
            raise Exception(except_info)


    def plot_record(self,fig_fpath):
        x = self.x
        fig= plt.figure(figsize=(8,3))
        # Befor EM updating
        ax1 = plt.subplot2grid((2,3),(0,0),rowspan=2)
        self.plot(ax1,x,self.iter_record['mu_all'][0],
                  self.iter_record['sigma_all'][0],
                  self.iter_record['pi_all'][0])
        ax1.set_title('Before EM')

        # After EM updating
        ax2 = plt.subplot2grid((2,3),(0,1),rowspan=2)
        self.plot(ax2,x,self.iter_record['mu_all'][-1],
                  self.iter_record['sigma_all'][-1],
                  self.iter_record['pi_all'][-1])
        ax2.set_title('After EM')

        # the function of lh about iteration
        ax3_1 = plt.subplot2grid((2,3),(0,2))
        ax3_1.plot(self.iter_record['lh'])
        ax3_2 = plt.subplot2grid((2,3),(1,2),sharex=ax3_1)
        ax3_2.plot(self.iter_record['lh'])

        lh_min = np.min(self.iter_record['lh'][1:])
        lh_max = np.max(self.iter_record['lh'][1:])
        ax3_1.set_ylim((lh_min,lh_max+(lh_max-lh_min)/10))
        ax3_2.set_ylim((self.iter_record['lh'][0]-(lh_max-lh_min)/10,
                           self.iter_record['lh'][0]+(lh_max-lh_min)/10))

        ax3_1.spines['bottom'].set_visible(False)
        ax3_2.spines['top'].set_visible(False)
        ax3_1.xaxis.tick_top()
        ax3_2.tick_params(labeltop=False)  # don't put tick labels at the top
        ax3_2.xaxis.tick_bottom()
        ax3_2.set_xlabel('iteration')
        ax3_1.set_title('likelihood')

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax3_1.transAxes, color='k', clip_on=False)
        ax3_1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax3_1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax3_2.transAxes)  # switch to the bottom axes
        ax3_2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax3_2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        plt.tight_layout()
        if fig_fpath is not None:
            fig.savefig(fig_fpath)
        plt.close(fig)

    def cal_prob(self,x):
        """
        """
        return self._gmm_pdf(x,self.mu_all,self.sigma_all,self.pi_all)


    def plot(self,ax,x,mu_all,sigma_all,pi_all):
        """ GMMs(contours) as well as data(scatter)
        Args:
            ax: axis
            x: numpy ndarray with shape of [n_sample, n_var]
            mu_all, sigma_all: parameters of each GM component
            pi: weights of each GM component
        Returns:
            list of line objects
        """
        line_all = []

        k = mu_all.shape[0]

        plot_tools.plot_line(ax,x[:,0],x[:,1],'bo',alpha=0.3,zorder=1,
                             markersize=3,line_container=line_all)

        n_point = 1000
        x0, x1 = np.meshgrid(np.linspace(-10, 10, n_point),
                             np.linspace(-10, 10, n_point))

        x0, x1 = np.meshgrid(np.linspace(np.min(x[:,0]),np.max(x[:,0]),n_point),
                             np.linspace(np.min(x[:,1]),np.max(x[:,1]),n_point))
        x0x1 = np.array([x0.flatten(),x1.flatten()]).T
        p = self._gmm_pdf(x0x1,mu_all,sigma_all,pi_all)
        p = p.reshape([x0.shape[0],x1.shape[0]])
        plot_tools.plot_contour(ax,x0,x1,p, levels=4, alpha=1,zorder=2,
                                line_container=line_all, colors='k')
        plot_tools.plot_line(ax,mu_all[:,0],mu_all[:,1], 'ro',
                             markersize=3, line_container=line_all,
                             zorder=3)
        return line_all


    def save(self,fpath):
        np.save(fpath,[self.mu_all,self.sigma_all,self.pi_all,
                       self.iter_record,self.lh_theta,self.max_iter])

    def load(self,fpath):
        [self.mu_all,self.sigma_all,self.pi_all,
         self.iter_record,self.lh_theta,self.max_iter] = np.load(fpath,allow_pickle=True)


class GMMs_classifier:
    def __init__(self):
        self.model_all = []
        self.label_all = []

    def load(self,model_fpath_all,label_all):
        self.model_all.clear()
        self.label_all.clear()
        for model_fpath,label in zip(model_fpath_all,label_all):
            gmms = GMMs()
            gmms.load(fpath=model_fpath)
            self.model_all.append(gmms)
            self.label_all.append(label)


    def classify(x):
        prob_all = np.asarray([model.cal_prob(x) for model in self.model_all])
        label_est = np.asarray([self.label_all[i]
                                    for i in np.argmax(prob_all,axis=0)])
        return label_est
