import numpy as np


def encode(state: np.ndarray, approach: str, full_state_idx) -> int | float:
    binary_str = reversed([str(cell) for cell in state])
    binary_str = "".join(binary_str)
    state = int(binary_str, 2)

    state = state / full_state_idx

    return state


def decode(state: int | float, n_bridges: int, full_state_idx: int) -> np.ndarray:
    if isinstance(state, float):
        state = int(state * full_state_idx)

    binary_str = bin(state)[2:].zfill(n_bridges)
    state = np.array([int(cell) for cell in reversed(binary_str)], dtype=np.int64)

    return state


def argmax(array: np.ndarray, action_rng: np.random.default_rng) -> int:
    try:
        return action_rng.choice(np.where(array == np.max(array))[0])
    except:
        return np.argmax(array)
