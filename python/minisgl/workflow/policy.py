from minisgl.core import Req

#####################
# Schedule Policies #
#####################

def sort_by_LPM(pending_list, **kwargs):
    if len(pending_list) <= 1:
        return pending_list
    
    cache_manager = kwargs.get("cache_manager")
    assert cache_manager is not None, "When use LPM scheduling, need to pass cache_manager argument"

    match_results = []
    for p_req in pending_list:
        if p_req.chunked_req:
            hit_len = p_req.chunked_req.cached_len
        else:
            handle = cache_manager.match_req(p_req).cuda_handle
            hit_len = handle.cached_len
        match_results.append((p_req, hit_len))

    match_results.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in match_results]

SCHEDULE_POLICY_MAP = {
    "LPM": sort_by_LPM,
}

##################
# Evict Policies #
##################

def LIFO(pending_list, **kwargs):
    # Last in, first out
    for i in range(len(pending_list) - 1, -1, -1):
        req = pending_list[i]
        if req.chunked_req is not None and type(req.chunked_req) is Req: # child req
            return i
    return None

EVICT_POLICY_MAP = {
    "LIFO": LIFO,
}