from tianshou.data import Batch
from typing import Any, Dict, Optional, Union
from tianshou.policy import BasePolicy
import numpy as np
from mao_env.mao import *

class UnoPolicy(BasePolicy):
    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any,) -> Batch:
        mask = batch.obs.mask[0]
        observation = MaoGame(Config(4,["Alpha","Beta","Gamma","Delta"],52)).unflatten_observation(batch.obs.obs[0])
        played_cards = observation['played_cards']
        if len(played_cards)==0:
            uno_mask = [False for _ in range(52)]
        else:
            uno_mask = [id_to_card(id).suit==played_cards[-1].suit or id_to_card(id).number==played_cards[-1].number for id in range(52)]
        logits = np.random.rand(*mask.shape)
        if (~np.logical_and(mask,uno_mask)).all():
            logits[~mask] = -np.inf
        else:
            logits[~np.logical_and(mask,uno_mask)] = -np.inf
        return Batch(act=[logits.argmax(axis=-1)], dtype=np.int64)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        return {}