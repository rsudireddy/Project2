import numpy as np

class BasicStump:
    """A single‐split regressor (depth‐1)."""
    def __init__(self):
        self.split_feature  = None
        self.split_value    = None
        self.output_low     = None
        self.output_high    = None

    def fit(self, data, labels):
        rows, cols = data.shape
        min_error = float('inf')
        feat_idx = 0
        while feat_idx < cols:
            unique_vals = np.unique(data[:, feat_idx])
            val_idx = 0
            while val_idx < unique_vals.size:
                thr = unique_vals[val_idx]
                mask_low  = data[:, feat_idx] <= thr
                mask_high = ~mask_low

                group_low  = labels[mask_low]
                group_high = labels[mask_high]

                low_mean  = group_low.mean()  if group_low.size  else 0
                high_mean = group_high.mean() if group_high.size else 0

                error = ((group_low - low_mean)**2).sum() + ((group_high - high_mean)**2).sum()
                if error < min_error:
                    min_error          = error
                    self.split_feature = feat_idx
                    self.split_value   = thr
                    self.output_low    = low_mean
                    self.output_high   = high_mean

                val_idx += 1
            feat_idx += 1

    def predict(self, data):
        n = data.shape[0]
        preds = np.empty(n, float)
        mask = data[:, self.split_feature] <= self.split_value

        # “switch” on True to assign both branches
        match True:
            case _:
                preds[mask]  = self.output_low
                preds[~mask] = self.output_high

        return preds


class BoostingClassifier:
    """
    Gradient boosting for binary classification.
    Default base learner is BasicStump (depth-1). Pass
    a different regressor class and params to use deeper trees.
    """
    def __init__(self,
                 n_rounds=100,
                 lr=0.1,
                 base_learner_cls=None,
                 base_learner_params=None):
        self.n_rounds      = n_rounds
        self.lr            = lr
        self.learners      = []
        self.init_logodds  = None

        match base_learner_cls:
            case None:
                from .GradientBoostingTree import BasicStump
                self.base_cls    = BasicStump
                self.base_params = {}
            case _:
                self.base_cls    = base_learner_cls
                self.base_params = base_learner_params or {}

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, data, labels):
        # initial log-odds
        p = np.clip(labels.mean(), 1e-6, 1 - 1e-6)
        self.init_logodds = np.log(p / (1 - p))
        F = np.full_like(labels, self.init_logodds, dtype=float)

        round_idx = 0
        while round_idx < self.n_rounds:
            # residual (gradient of log-loss)
            residuals = labels - self._sigmoid(F)

            learner = self.base_cls(**self.base_params)
            learner.fit(data, residuals)
            update = learner.predict(data)

            F += self.lr * update
            self.learners.append(learner)

            round_idx += 1

        return self

    def predict_proba(self, data):
        n = data.shape[0]
        F = np.full(n, self.init_logodds, dtype=float)
        for learner in self.learners:
            F += self.lr * learner.predict(data)
        return self._sigmoid(F)

    def predict(self, data):
        probs = self.predict_proba(data)
        n = probs.size
        preds = np.empty(n, int)

        # threshold at 0.5 with a match/case switch
        for i in range(n):
            match True:
                case _ if probs[i] >= 0.5:
                    preds[i] = 1
                case _:
                    preds[i] = 0

        return preds
