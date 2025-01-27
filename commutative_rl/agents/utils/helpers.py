import torch
import numpy as np

from typing import List, Union


def argmax(array: np.ndarray, action_rng: np.random.default_rng) -> int:
    try:
        return action_rng.choice(np.where(array == np.max(array))[0])
    except:
        return np.argmax(array)
