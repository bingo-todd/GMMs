import numpy as np
import os
import GMMs
from BasicTools.easy_parallel import easy_parallel


class GMMs_Classifier(object):
    """"""
    def __init__(self, model_dir, labels, n_band):

        n_label = len(labels)
        model_all = np.ndarray((n_label, n_band), dtype=np.object)
        print(f'load models from {model_dir}')
        for label_i, label in enumerate(labels):
            for band_i in range(n_band):
                model_fpath = os.path.join(model_dir, label, f'{band_i}.npy')
                model_all[label_i, band_i] = GMMs.GMMs()
                model_all[label_i, band_i].load(model_fpath)
        print('finish loading')

        self.n_band = n_band
        self.labels = labels
        self.model_all = model_all
        #
        self.epsilon = 1e-40

    def _cal_lh(self, model, x):
        return np.log(model.cal_prob(x) + self.epsilon)

    def cal_lh(self, x, n_worker=8):
        """
        x: [n_sample, n_band, fea_len]
        """
        n_label = len(self.labels)
        n_band = self.n_band

        # maximum likelihood
        n_sample = x.shape[0]
        lh = np.zeros((n_sample, n_label, n_band))
        tasks = []
        for label_i in range(n_label):
            for band_i in range(n_band):
                if True:
                    tasks.append([self.model_all[label_i, band_i],
                        x[:, band_i, :]])
        results = easy_parallel(self._cal_lh, tasks, n_worker=n_worker)
        count_tmp = 0
        for label_i in range(n_label):
            for band_i in range(n_band):
                lh[:, label_i, band_i] = results[count_tmp]
                count_tmp = count_tmp + 1
         
        # not parallel
        # for label_i in range(n_label):
        #     for band_i in range(n_band):
                # model_tmp = self.model_all[label_i, band_i]
                # posterior = model_tmp.cal_prob(x[:, band_i, :])
                # lh[:, label_i, band_i] = np.log(posterior+self.epsilon)
                # itd_flag_frame_all = np.abs(x[:, band_i, 0]) == theta_itd
                # lh[itd_flag_frame_all, label_i, band_i] = 0
        # prob of bands are multipled together

        lh = np.sum(lh, axis=2)
        return lh

    def cal_lh_band_all(self, x):
        """
        x: [n_sample, n_band, fea_len]
        """
        n_label = len(self.labels)
        n_band = self.n_band
        n_sample = x.shape[0]
        lh = np.zeros((n_sample, n_label, n_band))
        for label_i in range(n_label):
            for band_i in range(n_band):
                model_tmp = self.model_all[label_i, band_i]
                lh[:, label_i, band_i] = model_tmp.cal_prob(x[:, band_i, :])
        return lh

    def classify(self, x):
        """
        Args:
            x: input features, [n_sample, n_band, fea_len]
        """
        lh = self.cal_lh(x)
        est_index = np.argmax(lh, axis=1)
        return [self.labels[i] for i in est_index]

    def estimate(self, x):
        """
        Args:
            x: input features, [n_sample, n_band, fea_len]
        """
        lh = self.cal_lh(x)
        y_est = self._get_max_pos(lh)
        return y_est

    def _get_max_pos(self, lh):
        n_sample, lh_len = lh.shape
        max_pos_all = np.zeros(n_sample)
        for sample_i in range(n_sample):
            lh_sample = lh[sample_i]
            lh_sample = lh_sample-np.min(lh_sample)+1.0  # square should >2
            max_pos_tmp = np.argmax(lh_sample)
            if max_pos_tmp >= 1 and max_pos_tmp <= lh_sample.shape[0]-2:
                [value_l, value_m,
                 value_r] = np.log(lh_sample[max_pos_tmp-1:max_pos_tmp+2])
                delta = (value_r-value_l)/(2*(2*value_m-value_l-value_r))
            else:
                delta = 0
            max_pos_all[sample_i] = max_pos_tmp+delta
        return max_pos_all
