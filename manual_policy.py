from tianshou.data import Batch
from typing import Any, Dict, Optional, Union
from tianshou.policy import BasePolicy
import numpy as np
from mao_env.mao import *

game_copy = MaoGame(Config(4,["Alpha","Beta","Gamma","Delta"],52))

class UnoPolicy(BasePolicy):
    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any,) -> Batch:
        mask = batch.obs.mask
        observation = [game_copy.unflatten_observation(obs) for obs in batch.obs.obs]
        batched_logits = []
        for i in range(len(observation)):
            played_cards = observation[i]['played_cards']
            if len(played_cards)==0:
                uno_mask = [False for _ in range(52)]
            else:
                uno_mask = [id_to_card(id).suit==played_cards[-1].suit or id_to_card(id).number==played_cards[-1].number for id in range(52)]
            logits = np.random.rand(52)
            if (~np.logical_and(mask[i],uno_mask)).all():
                logits[~mask[i]] = -np.inf
            else:
                logits[~np.logical_and(mask[i],uno_mask)] = -np.inf
            batched_logits.append(logits)
        return Batch(act=np.array(batched_logits).argmax(axis=-1), dtype=np.int64)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        return {}