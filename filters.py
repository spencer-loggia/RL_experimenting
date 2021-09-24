import numpy as np

def partial_observability_filter(global_state, observe_dist, origin):
    h = global_state.shape[0]
    w = global_state.shape[1]
    if observe_dist*2 >= h or observe_dist*2 >= w:
        raise IndexError
    lb = max(origin[1]-observe_dist, 0)
    rb = min(origin[1]+observe_dist, w)
    tb = max(origin[0]-observe_dist, 0)
    bb = min(origin[0]+observe_dist, h)
    partial_dist = global_state[tb:bb, lb:rb]
    pw = partial_dist.shape[1]
    ph = partial_dist.shape[0]
    offsetw = observe_dist*2 - pw
    if offsetw > 0 and lb == 0:
        fill = np.zeros((ph, offsetw))
        partial_dist = np.concatenate([fill, partial_dist], axis=1)
    elif offsetw > 0 and rb == w:
        fill = np.zeros((ph, offsetw))
        partial_dist = np.concatenate([partial_dist, fill], axis=1)
    pw = partial_dist.shape[1]
    ph = partial_dist.shape[0]
    offseth = observe_dist * 2 - ph
    if offseth > 0 and tb == 0:
        fill = np.zeros((offseth, pw))
        partial_dist = np.concatenate([fill, partial_dist], axis=0)
    elif offseth > 0 and bb == h:
        fill = np.zeros((offseth, pw))
        partial_dist = np.concatenate([partial_dist, fill], axis=0)
    return partial_dist
