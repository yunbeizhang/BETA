"""
Expected Calibration Error (ECE) and related calibration losses.

Lightweight, NumPy-only implementation of the metrics used in the paper to
report Expected Calibration Error alongside top-1 accuracy. Other calibration
variants (MCE, SCE, TACE) are kept here as drop-in references for future use.
"""

import numpy as np
from scipy.special import softmax


class CELoss:
    """Shared bookkeeping for calibration losses based on bin statistics."""

    def compute_bin_boundaries(self, probabilities=np.array([])):
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            bin_n = int(self.n_data / self.n_bins)
            bin_boundaries = np.array([])
            probabilities_sort = np.sort(probabilities)
            for i in range(self.n_bins):
                bin_boundaries = np.append(bin_boundaries, probabilities_sort[i * bin_n])
            bin_boundaries = np.append(bin_boundaries, 1.0)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]

    def get_probabilities(self, output, labels, logits):
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.predictions = np.argmax(self.probabilities, axis=1)
        self.accuracies = np.equal(self.predictions, labels)

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        pred_matrix = np.zeros([self.n_data, self.n_class])
        label_matrix = np.zeros([self.n_data, self.n_class])
        pred_matrix[idx, self.predictions] = 1
        label_matrix[idx, self.labels] = 1
        self.acc_matrix = np.equal(pred_matrix, label_matrix)

    def compute_bins(self, index=None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index is None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:, index]
            accuracies = self.acc_matrix[:, index]

        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = np.greater(confidences, bin_lower.item()) * np.less_equal(
                confidences, bin_upper.item()
            )
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                if index is None:
                    self.bin_acc[i] = np.mean(accuracies[in_bin])
                else:
                    self.bin_acc[i] = np.mean(np.equal(self.labels[in_bin], index))
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])


class MaxProbCELoss(CELoss):
    def loss(self, output, labels, n_bins=15, logits=True):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()


class ECELoss(MaxProbCELoss):
    """Expected Calibration Error over `n_bins` equal-width confidence bins."""

    def loss(self, output, labels, n_bins=15, logits=True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop, self.bin_score)


class MCELoss(MaxProbCELoss):
    def loss(self, output, labels, n_bins=15, logits=True):
        super().loss(output, labels, n_bins, logits)
        return np.max(self.bin_score)
