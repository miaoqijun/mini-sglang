from dataclasses import dataclass, field
from typing import List
import bisect

from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.utils import PendingReq
from .psrt import PSRTNode

@dataclass
class PSRTPrefillManager(PrefillManager):
    priority_list: List[int] = field(default_factory=list)

    def add_one_req(self, uid, input_ids, sampling_params, priority) -> None:
        idx = bisect.bisect_right(self.priority_list, priority)
        self.priority_list.insert(idx, priority)
        self.pending_list.insert(idx, PendingReq(uid, input_ids, sampling_params))