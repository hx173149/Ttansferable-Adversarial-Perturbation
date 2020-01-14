from abc import ABCMeta
import numpy as np
from six.moves import xrange
import warnings


class Attack(object):

    """
    Abstract base class for all attack classes.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, back='tf', sess=None):
        """
        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's predictions.
        :param back: The backend to use. Either 'tf' (default) or 'th'.
        :param sess: The tf session to run graphs in (use None for Theano)
        """
        if not(back == 'tf' or back == 'th'):
            raise ValueError("Backend argument must either be 'tf' or 'th'.")
        if back == 'th' and sess is not None:
            raise Exception("A session should not be provided when using th.")
        if not hasattr(model, '__call__'):
            raise ValueError("model argument must be a function that returns "
                             "the symbolic output when given an input tensor.")
        if back == 'th':
            warnings.warn("CleverHans support for Theano is deprecated and "
                          "will be dropped on 2017-11-08.")

        # Prepare attributes
        self.model = model
        self.back = back
        self.sess = sess
        self.inf_loop = False

    def generate(self, x, **kwargs):
        """
        Generate the attack's symbolic graph for adversarial examples. This
        method should be overriden in any child class that implements an
        attack that is expressable symbolically. Otherwise, it will wrap the
        numerical implementation as a symbolic operator.
        :param x: The model's symbolic inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A symbolic representation of the adversarial examples.
        """
        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')

        if not self.inf_loop:
            self.inf_loop = True
            assert self.parse_params(**kwargs)
            import tensorflow as tf
            graph = tf.py_func(self.generate_np, [x], tf.float32)
            self.inf_loop = False
            return graph
        else:
            error = "No symbolic or numeric implementation of attack."
            raise NotImplementedError(error)

    def generate_np(self, x_val, **kwargs):
        """
        Generate adversarial examples and return them as a Numpy array. This
        method should be overriden in any child class that implements an attack
        that is not fully expressed symbolically.
        :param x_val: A Numpy array with the original inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A Numpy array holding the adversarial examples.
        """
        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')

        if not self.inf_loop:
            self.inf_loop = True
            import tensorflow as tf

            # Generate this attack's graph if not done previously
            if not hasattr(self, "_x") and not hasattr(self, "_x_adv"):
                input_shape = list(x_val.shape)
                input_shape[0] = None
                self._x = tf.placeholder(tf.float32, shape=input_shape)
                self._x_adv = self.generate(self._x, **kwargs)
            self.inf_loop = False
        else:
            error = "No symbolic or numeric implementation of attack."
            raise NotImplementedError(error)

        if self.sess is None:
            raise ValueError("Cannot use `generate_np` when no `sess` was"
                             " provided")
        return self.sess.run(self._x_adv, feed_dict={self._x: x_val})

    def parse_params(self, params=None):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        :param params: a dictionary of attack-specific parameters
        :return: True when parsing was successful
        """
        return True


class FastGradientMethod(Attack):

    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm (and is known as the "Fast Gradient Sign Method"). This
    implementation extends the attack to other norms, and is therefore called
    the Fast Gradient Method.
    Paper link: https://arxiv.org/abs/1412.6572
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a FastGradientMethod instance.
        """
        super(FastGradientMethod, self).__init__(model, back, sess)
    
    def generate_with_feats_diff(self, x_feat,x,last_grad,feat_scale,conv_scale,pow_scale,**kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        from .attacks_tf import fgm_all_feats

        import tensorflow as tf
        tf.stop_gradient(x)
        adv_x,last_grad = fgm_all_feats(x_feat,x,last_grad,self.model(x_feat),
                self.model.all_feats(x_feat),self.model.all_feats(x),
                feat_scale,conv_scale,pow_scale,y=self.y, eps=self.eps, ord=self.ord,
                   clip_min=self.clip_min, clip_max=self.clip_max)
        return adv_x,last_grad
    
    def parse_params(self, eps=0.3, ord=np.inf, y=None, clip_min=None,
                     clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters
        self.eps = eps
        self.ord = ord
        self.y = y
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.back == 'th' and self.ord != np.inf:
            raise NotImplementedError("The only FastGradientMethod norm "
                                      "implemented for Theano is np.inf.")
        return True


class OwnMethod(Attack):

    def __init__(self, model, back='tf', sess=None):
        """
        Create a OwnMethod instance.
        """
        super(OwnMethod, self).__init__(model,back, sess)

    def generate(self, x, feat_scale,conv_scale,pow_scale,**kwargs):
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = 0
        #eta = tf.random_normal(x.get_shape()) * 1e-5 
        last_grad = None

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        y_default = tf.to_float(tf.equal(model_preds, preds_max))
        fgsm_params = {'eps': self.eps_iter, 'y': y_default, 'ord': self.ord}
        FGSM = FastGradientMethod(self.model, back=self.back,
                sess=self.sess)
        for i in range(self.nb_iter):
            # Compute this step's perturbation
            new_x,last_grad = FGSM.generate_with_feats_diff(x+eta,x,last_grad,feat_scale,conv_scale,pow_scale,**fgsm_params)
            eta = new_x-x
            
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm

        # Define adversarial example (and clip if necessary)
        return tf.sign(eta)*self.eps

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (required) A placeholder for the model labels.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.back == 'th':
            error_string = "OwnMethod is not implemented in Theano"
            raise NotImplementedError(error_string)

        return True
