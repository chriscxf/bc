import torch
import matplotlib.pyplot as plt
from scipy.stats import t
import math
import numpy as np
import pandas as pd
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F

from _basicClasses import TimeSeriesDatasetSingleSubject





def joint_likelihood(d, y, mu, sigma, p):

    #  Parameters:
    # - mu: Predicted means for each time point, shape [batch_size, predict_time_window].
    # - sigma: Predicted standard deviations for each time point, shape [batch_size, predict_time_window].
    # - p: Predicted probabilities for the binary event, shape [batch_size, predict_time_window].
    # - y: True values, shape [batch_size, predict_time_window].
    # - d: Binary event labels, shape [batch_size, predict_time_window].
    # - eps: A small value to avoid log of zero errors.

    # Returns:
    # - The average loss per time point across all batches and all predicted time points.
  

    eps = 1e-8  # Small epsilon to prevent division by zero, not used for now

    gaussian_loss = 0.5 * torch.log(2 * math.pi * sigma**2 + eps) + ((y - mu) ** 2) / (2 * (sigma ** 2))

    # Binary cross-entropy part for event
    bce_loss = -d * torch.log(p + eps) - (1 - d) * torch.log(1 - p + eps)

 
    return (gaussian_loss+bce_loss).mean()

