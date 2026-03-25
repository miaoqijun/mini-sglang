from minisgl.core import Req


def _get_hit_len(p_req, cache_manager):
    if p_req.chunked_req:
        return p_req.chunked_req.cached_len
    handle = cache_manager.match_req(p_req).cuda_handle
    return handle.cached_len


def _get_extend_len(p_req, cache_manager):
    return p_req.input_len - _get_hit_len(p_req, cache_manager)


def _iter_child_reqs(pending_list):
    for i, req in enumerate(pending_list):
        if req.chunked_req is not None and type(req.chunked_req) is Req:  # child req
            yield i, req


def _get_node(req, uid2node):
    if uid2node is None:
        return None
    return uid2node.get(req.uid)


def _get_root_node(node, uid2node):
    if node is None or uid2node is None:
        return None
    return uid2node.get(node.get_root())


def _get_root_age(req, uid2node):
    node = _get_node(req, uid2node)
    root = _get_root_node(node, uid2node)
    if root is None or root.t is None:
        return 0
    return -root.t


def _get_subtree_size(req, uid2node):
    node = _get_node(req, uid2node)
    if node is None:
        return 1
    return node.get_size()


def _get_critical_depth_node(node):
    if node is None or len(node.children) == 0:
        return 0
    return 1 + max(_get_critical_depth_node(child) for child in node.children)


def _get_critical_depth(req, uid2node):
    node = _get_node(req, uid2node)
    return _get_critical_depth_node(node)


def _get_logical_inherited_len(req, status_map):
    if status_map is None or req.uid not in status_map:
        return 0
    return status_map[req.uid].inherited_len


def _normalize_values(values):
    if len(values) == 0:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [0.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]

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
        hit_len = _get_hit_len(p_req, cache_manager)
        match_results.append((p_req, hit_len))

    match_results.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in match_results]


def sort_by_LPM_SEF(pending_list, **kwargs):
    if len(pending_list) <= 1:
        return pending_list

    cache_manager = kwargs.get("cache_manager")
    assert cache_manager is not None, "When use LPM_SEF scheduling, need to pass cache_manager argument"

    scored = []
    for p_req in pending_list:
        hit_len = _get_hit_len(p_req, cache_manager)
        extend_len = p_req.input_len - hit_len
        scored.append((p_req, hit_len, extend_len))

    scored.sort(key=lambda x: (-x[1], x[2]))
    return [x[0] for x in scored]


def sort_by_SEF(pending_list, **kwargs):
    if len(pending_list) <= 1:
        return pending_list

    cache_manager = kwargs.get("cache_manager")
    assert cache_manager is not None, "When use SEF scheduling, need to pass cache_manager argument"

    scored = []
    for p_req in pending_list:
        extend_len = _get_extend_len(p_req, cache_manager)
        scored.append((p_req, extend_len))

    scored.sort(key=lambda x: x[1])
    return [x[0] for x in scored]


def sort_by_COMPLETE_REUSE(pending_list, **kwargs):
    if len(pending_list) <= 1:
        return pending_list

    cache_manager = kwargs.get("cache_manager")
    uid2node = kwargs.get("uid2node")
    assert cache_manager is not None, "When use COMPLETE_REUSE scheduling, need to pass cache_manager argument"

    scored = []
    for p_req in pending_list:
        hit_len = _get_hit_len(p_req, cache_manager)
        extend_len = _get_extend_len(p_req, cache_manager)
        subtree_size = _get_subtree_size(p_req, uid2node)
        root_age = _get_root_age(p_req, uid2node)
        scored.append((p_req, hit_len, subtree_size, extend_len, root_age))

    # Prefer high reuse, then branches/workflows closer to completion,
    # then smaller incremental work, then older roots for fairness.
    scored.sort(key=lambda x: (-x[1], x[2], x[3], x[4]))
    return [x[0] for x in scored]


def sort_by_CRITICAL_REUSE(pending_list, **kwargs):
    if len(pending_list) <= 1:
        return pending_list

    cache_manager = kwargs.get("cache_manager")
    uid2node = kwargs.get("uid2node")
    assert cache_manager is not None, "When use CRITICAL_REUSE scheduling, need to pass cache_manager argument"

    scored = []
    for p_req in pending_list:
        critical_depth = _get_critical_depth(p_req, uid2node)
        hit_len = _get_hit_len(p_req, cache_manager)
        extend_len = _get_extend_len(p_req, cache_manager)
        subtree_size = _get_subtree_size(p_req, uid2node)
        scored.append((p_req, critical_depth, hit_len, extend_len, subtree_size))

    # Prefer nodes on deeper critical paths, but still reward high reuse
    # and lower incremental prefill cost.
    scored.sort(key=lambda x: (-x[1], -x[2], x[3], x[4]))
    return [x[0] for x in scored]


def sort_by_REUSE_DENSITY(pending_list, **kwargs):
    if len(pending_list) <= 1:
        return pending_list

    cache_manager = kwargs.get("cache_manager")
    uid2node = kwargs.get("uid2node")
    assert cache_manager is not None, "When use REUSE_DENSITY scheduling, need to pass cache_manager argument"

    scored = []
    for p_req in pending_list:
        hit_len = _get_hit_len(p_req, cache_manager)
        extend_len = _get_extend_len(p_req, cache_manager)
        density = hit_len / max(extend_len, 1)
        subtree_size = _get_subtree_size(p_req, uid2node)
        scored.append((p_req, density, hit_len, subtree_size, extend_len))

    scored.sort(key=lambda x: (-x[1], -x[2], x[3], x[4]))
    return [x[0] for x in scored]


def sort_by_WEIGHTED_SCORE(pending_list, **kwargs):
    if len(pending_list) <= 1:
        return pending_list

    cache_manager = kwargs.get("cache_manager")
    uid2node = kwargs.get("uid2node")
    assert cache_manager is not None, "When use WEIGHTED_SCORE scheduling, need to pass cache_manager argument"

    reqs = list(pending_list)
    hit_lens = [_get_hit_len(p_req, cache_manager) for p_req in reqs]
    extend_lens = [_get_extend_len(p_req, cache_manager) for p_req in reqs]
    subtree_sizes = [_get_subtree_size(p_req, uid2node) for p_req in reqs]
    critical_depths = [_get_critical_depth(p_req, uid2node) for p_req in reqs]

    norm_hit_lens = _normalize_values(hit_lens)
    norm_extend_lens = _normalize_values(extend_lens)
    norm_subtree_sizes = _normalize_values(subtree_sizes)
    norm_critical_depths = _normalize_values(critical_depths)

    scored = []
    for i, p_req in enumerate(reqs):
        score = (
            0.45 * norm_hit_lens[i]
            - 0.20 * norm_extend_lens[i]
            - 0.20 * norm_subtree_sizes[i]
            + 0.15 * norm_critical_depths[i]
        )
        scored.append((p_req, score, hit_lens[i], extend_lens[i]))

    scored.sort(key=lambda x: (-x[1], -x[2], x[3]))
    return [x[0] for x in scored]

SCHEDULE_POLICY_MAP = {
    "LPM": sort_by_LPM,
    "LPM_SEF": sort_by_LPM_SEF,
    "SEF": sort_by_SEF,
    "COMPLETE_REUSE": sort_by_COMPLETE_REUSE,
    "CRITICAL_REUSE": sort_by_CRITICAL_REUSE,
    "REUSE_DENSITY": sort_by_REUSE_DENSITY,
    "WEIGHTED_SCORE": sort_by_WEIGHTED_SCORE,
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


def SMALLEST_CACHE(pending_list, **kwargs):
    victim_idx = None
    victim_cached = None
    for i, req in _iter_child_reqs(pending_list):
        cached_len = req.chunked_req.cached_len
        if victim_cached is None or cached_len < victim_cached:
            victim_idx = i
            victim_cached = cached_len
    return victim_idx


def LARGEST_REMAIN(pending_list, **kwargs):
    victim_idx = None
    victim_remain = None
    for i, req in _iter_child_reqs(pending_list):
        remain = req.input_len - req.chunked_req.cached_len
        if victim_remain is None or remain > victim_remain:
            victim_idx = i
            victim_remain = remain
    return victim_idx


def SMALLEST_SUBTREE(pending_list, **kwargs):
    uid2node = kwargs.get("uid2node")
    if uid2node is None:
        return None

    victim_idx = None
    victim_size = None
    for i, req in _iter_child_reqs(pending_list):
        node = uid2node.get(req.uid)
        if node is None:
            continue
        subtree_size = node.get_size()
        if victim_size is None or subtree_size < victim_size:
            victim_idx = i
            victim_size = subtree_size
    return victim_idx


def LEAF_FIRST(pending_list, **kwargs):
    uid2node = kwargs.get("uid2node")

    victim_idx = None
    victim_score = None
    for i, req in _iter_child_reqs(pending_list):
        critical_depth = _get_critical_depth(req, uid2node)
        subtree_size = _get_subtree_size(req, uid2node)
        score = (critical_depth, subtree_size, req.chunked_req.cached_len)
        if victim_score is None or score < victim_score:
            victim_idx = i
            victim_score = score
    return victim_idx


def FAR_AND_CHEAP(pending_list, **kwargs):
    uid2node = kwargs.get("uid2node")

    victim_idx = None
    victim_score = None
    for i, req in _iter_child_reqs(pending_list):
        critical_depth = _get_critical_depth(req, uid2node)
        subtree_size = _get_subtree_size(req, uid2node)
        cached_len = req.chunked_req.cached_len
        # Prefer evicting requests that are far from completion (large subtree /
        # deeper path) but currently have little physically cached prefix.
        score = (critical_depth + subtree_size, -cached_len)
        if victim_score is None or score > victim_score:
            victim_idx = i
            victim_score = score
    return victim_idx


def LOW_PRIORITY(pending_list, **kwargs):
    cache_manager = kwargs.get("cache_manager")
    uid2node = kwargs.get("uid2node")
    if cache_manager is None:
        return None

    victim_idx = None
    victim_score = None
    for i, req in _iter_child_reqs(pending_list):
        hit_len = _get_hit_len(req, cache_manager)
        extend_len = _get_extend_len(req, cache_manager)
        subtree_size = _get_subtree_size(req, uid2node)
        root_age = _get_root_age(req, uid2node)
        score = (hit_len, -subtree_size, -extend_len, root_age)
        if victim_score is None or score < victim_score:
            victim_idx = i
            victim_score = score
    return victim_idx


def WEIGHTED_EVICT(pending_list, **kwargs):
    uid2node = kwargs.get("uid2node")
    status_map = kwargs.get("status_map")

    child_items = list(_iter_child_reqs(pending_list))
    if len(child_items) == 0:
        return None

    cached_lens = [req.chunked_req.cached_len for _, req in child_items]
    remain_lens = [req.input_len - req.chunked_req.cached_len for _, req in child_items]
    subtree_sizes = [_get_subtree_size(req, uid2node) for _, req in child_items]
    critical_depths = [_get_critical_depth(req, uid2node) for _, req in child_items]
    page_wastes = [
        max(_get_logical_inherited_len(req, status_map) - req.chunked_req.cached_len, 0)
        for _, req in child_items
    ]

    norm_cached_lens = _normalize_values(cached_lens)
    norm_remain_lens = _normalize_values(remain_lens)
    norm_subtree_sizes = _normalize_values(subtree_sizes)
    norm_critical_depths = _normalize_values(critical_depths)
    norm_page_wastes = _normalize_values(page_wastes)

    victim_idx = None
    victim_score = None
    for i, (pending_idx, req) in enumerate(child_items):
        score = (
            0.35 * (1.0 - norm_cached_lens[i])
            + 0.25 * norm_remain_lens[i]
            + 0.15 * (1.0 - norm_subtree_sizes[i])
            + 0.15 * (1.0 - norm_critical_depths[i])
            + 0.10 * norm_page_wastes[i]
        )
        if victim_score is None or score > victim_score:
            victim_idx = pending_idx
            victim_score = score
    return victim_idx

EVICT_POLICY_MAP = {
    "LIFO": LIFO,
    "SMALLEST_CACHE": SMALLEST_CACHE,
    "LARGEST_REMAIN": LARGEST_REMAIN,
    "SMALLEST_SUBTREE": SMALLEST_SUBTREE,
    "LEAF_FIRST": LEAF_FIRST,
    "FAR_AND_CHEAP": FAR_AND_CHEAP,
    "LOW_PRIORITY": LOW_PRIORITY,
    "WEIGHTED_EVICT": WEIGHTED_EVICT,
}
