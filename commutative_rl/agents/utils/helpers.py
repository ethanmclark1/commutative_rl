import torch
import numpy as np


def encode(input_value: int, max_val: int, to_tensor: bool = False) -> float:
    encoded = input_value / max_val

    if to_tensor:
        encoded = torch.as_tensor(encoded, dtype=torch.float32).view(-1)

    return encoded


def argmax(array: np.ndarray, action_rng: np.random.default_rng) -> int:
    try:
        return action_rng.choice(np.where(array == np.max(array))[0])
    except:
        return np.argmax(array)
