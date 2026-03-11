from minisgl.scheduler.prefill import PrefillManager
from minisgl.core import Batch

class LPMPrefillManager(PrefillManager):
    def _sort_by_lpm(self):
        if len(self.pending_list) <= 1:
            return

        match_results = []
        for p_req in self.pending_list:
            if p_req.chunked_req:
                hit_len = p_req.chunked_req.cached_len
            else:
                handle = self.cache_manager.match_req(p_req).cuda_handle
                hit_len = handle.cached_len
            match_results.append((p_req, hit_len))

        match_results.sort(key=lambda x: x[1], reverse=True)
        self.pending_list = [x[0] for x in match_results]

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        self._sort_by_lpm()
        
        return super().schedule_next_batch(prefill_budget)