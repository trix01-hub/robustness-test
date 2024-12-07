import pickle
import numpy as np
import tensorflow as tf
import os

class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim, kl_targ, model_path, save_x_episode_model, seed, divider):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
        """

        self.divider = divider
        self.seed = seed
        self.save_x_episode_model = save_x_episode_model
        self.model_path = model_path
        self.graph_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../graph'))
        self.episodes = 0
        self.all_steps_remainder = 0
        self.all_steps = 0

        if(os.path.exists(self.model_path + '/info/policy.pkl')):
            policy_file = open(self.model_path + "/info/policy.pkl", "rb")
            policy_data = pickle.load(policy_file)
            policy_file.close()
            self.beta = policy_data['beta']
            self.lr_multiplier = policy_data['lr_multiplier']
        else:
            self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
            self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.lr = None
        self.kl_targ = kl_targ
        self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.epochs = 20
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.compat.v1.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
            self.init = tf.compat.v1.global_variables_initializer()
            self.saver = tf.compat.v1.train.Saver(max_to_keep=1000) 

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.beta_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (), 'beta')
        self.eta_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (), 'eta')
        # learning rate:
        self.lr_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (), 'eta')
        # log_vars and means with pi_old (previous step's policy parameters):
        self.old_log_vars_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_size = self.obs_dim * 10  # 10 empirically determined
        hid3_size = self.act_dim * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
        # 3 hidden layers with tanh activations
        out = tf.compat.v1.layers.dense(self.obs_ph, hid1_size, tf.compat.v1.tanh,
                              kernel_initializer=tf.compat.v1.random_normal_initializer(
                                  seed=self.seed, stddev=np.sqrt(1 / self.obs_dim)), name="h1")
        out = tf.compat.v1.layers.dense(out, hid2_size, tf.compat.v1.tanh,
                              kernel_initializer=tf.compat.v1.random_normal_initializer(
                                  seed=self.seed, stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.compat.v1.layers.dense(out, hid3_size, tf.compat.v1.tanh,
                              kernel_initializer=tf.compat.v1.random_normal_initializer(
                                  seed=self.seed, stddev=np.sqrt(1 / hid2_size)), name="h3")
        self.means = tf.compat.v1.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.compat.v1.random_normal_initializer(
                                         seed=self.seed, stddev=np.sqrt(1 / hid3_size)), name="means")
        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_size) // 48
        log_vars = tf.compat.v1.get_variable('logvars', (logvar_speed, self.act_dim), tf.compat.v1.float32,
                                   tf.compat.v1.constant_initializer(0.0))
        self.log_vars = tf.compat.v1.reduce_sum(log_vars, axis=0) - 1.0

        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_size, self.lr, logvar_speed))

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.compat.v1.reduce_sum(self.log_vars)
        logp += -0.5 * tf.compat.v1.reduce_sum(tf.compat.v1.square(self.act_ph - self.means) /
                                     tf.compat.v1.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.compat.v1.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.compat.v1.reduce_sum(tf.compat.v1.square(self.act_ph - self.old_means_ph) /
                                         tf.compat.v1.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _kl_entropy(self):
        """
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        log_det_cov_old = tf.compat.v1.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.compat.v1.reduce_sum(self.log_vars)
        tr_old_new = tf.compat.v1.reduce_sum(tf.compat.v1.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.compat.v1.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.compat.v1.reduce_sum(tf.compat.v1.square(self.means - self.old_means_ph) /
                                                     tf.compat.v1.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.compat.v1.reduce_sum(self.log_vars))

    def _sample(self):
        """ Sample from distribution, given observation """
        self.sampled_act = self.means + tf.math.exp(self.log_vars / 2.0) * tf.random.normal(shape=(self.act_dim,), seed=self.seed)

    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        loss1 = -tf.compat.v1.reduce_mean(self.advantages_ph *
                                tf.compat.v1.exp(self.logp - self.logp_old))
        loss2 = tf.compat.v1.reduce_mean(self.beta_ph * self.kl)
        loss3 = self.eta_ph * tf.compat.v1.square(tf.compat.v1.maximum(0.0, self.kl - 2.0 * self.kl_targ))
        self.loss = loss1 + loss2 + loss3
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.compat.v1.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, logger, all_steps):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                  feed_dict={self.obs_ph: observes})

        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier,
                     self.old_log_vars_ph: old_log_vars_np,
                     self.old_means_ph: old_means_np}

        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        self.all_steps = all_steps
        if(self.all_steps_remainder < all_steps//self.divider):
            self.all_steps_remainder = all_steps//self.divider
            with open(self.model_path + '/' + str(all_steps) + '/info/episodes.txt', 'w') as f:
                f.write(str(all_steps))
            self.save_policy()

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier})


    def save_policy(self):
        policy_data = {"beta": self.beta, "lr_multiplier": self.lr_multiplier}

        self.saver.save(self.sess, self.model_path + '/' + str(self.all_steps) + '/model', global_step=self.all_steps)
        with open(self.model_path + '/' + str(self.all_steps) + "/info/policy.pkl", "wb") as f:
                pickle.dump(policy_data, f)


    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

