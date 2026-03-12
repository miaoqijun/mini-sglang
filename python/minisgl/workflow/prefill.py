from dataclasses import dataclass
from typing import List, Tuple
import bisect

from minisgl.scheduler.prefill import PrefillManager
from .psrt import PSRTNode

@dataclass
class PSRTPrefillManager(PrefillManager):
    pending_list: List[Tuple(int, PendingReq)] = field(default_factory=list)

    def add_one_req(self, uid, input_ids, sampling_params, priority) -> None:
        keys = [k for _, k in self.pending_list]
        idx = bisect.bisect_left(keys, priority)
        self.pending_list.insert(idx, (PendingReq(uid, input_ids, sampling_params), priority))