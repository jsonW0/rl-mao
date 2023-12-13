from tianshou.data import Batch
from typing import Any, Dict, Optional, Union
from tianshou.policy import BasePolicy
import numpy as np
from mao_env.mao import *

class ManualPolicy(BasePolicy):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.game_copy = MaoGame(self.config)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any,) -> Batch:
        mask = batch.obs.mask
        observation = batch.obs.obs # no need to unflattetn
        batched_logits = []
        for i in range(len(observation)):
            rule_mask = []
            for id in range(52):
                potential_card = id_to_card(id)
                rule_mask.append(True)
                self.game_copy.played_cards = [id_to_card(observation[i][0])] if observation[i][0]!=52 else []
                for rule in self.config.validity_rules:
                    if rule(self.game_copy,potential_card):
                        pass
                    else:
                        rule_mask[-1] = False
            
            # top_card = observation[i][0]
            # if top_card == 52: # No top card, game just started
            #     uno_mask = [False for _ in range(52)]
            # else:
            #     uno_mask = [id_to_card(id).suit==id_to_card(top_card).suit or id_to_card(id).number==id_to_card(top_card).number for id in range(52)]
            logits = np.random.rand(52)
            if (~np.logical_and(mask[i],rule_mask)).all():
                logits[~mask[i]] = -np.inf
            else:
                logits[~np.logical_and(mask[i],rule_mask)] = -np.inf
            batched_logits.append(logits)
        return Batch(act=np.array(batched_logits).argmax(axis=-1), dtype=np.int64)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        return {}