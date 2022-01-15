import numpy as np
from scipy.ndimage import zoom, rotate
from scipy.spatial.distance import cdist
import torch


# def partial_observability_filter(global_state, observe_dist, origin):
#     h = global_state.shape[0]
#     w = global_state.shape[1]
#     if observe_dist*2 >= h or observe_dist*2 >= w:
#         raise IndexError
#     lb = max(origin[1]-observe_dist, 0)
#     rb = min(origin[1]+observe_dist, w)
#     tb = max(origin[0]-observe_dist, 0)
#     bb = min(origin[0]+observe_dist, h)
#     partial_dist = global_state[tb:bb, lb:rb]
#     pw = partial_dist.shape[1]
#     ph = partial_dist.shape[0]
#     offsetw = observe_dist*2 - pw
#     if offsetw > 0 and lb == 0:
#         fill = np.zeros((ph, offsetw))
#         partial_dist = np.concatenate([fill, partial_dist], axis=1)
#     elif offsetw > 0 and rb == w:
#         fill = np.zeros((ph, offsetw))
#         partial_dist = np.concatenate([partial_dist, fill], axis=1)
#     pw = partial_dist.shape[1]
#     ph = partial_dist.shape[0]
#     offseth = observe_dist * 2 - ph
#     if offseth > 0 and tb == 0:
#         fill = np.zeros((offseth, pw))
#         partial_dist = np.concatenate([fill, partial_dist], axis=0)
#     elif offseth > 0 and bb == h:
#         fill = np.zeros((offseth, pw))
#         partial_dist = np.concatenate([partial_dist, fill], axis=0)
#     return partial_dist


def vision_sim(pid, board_state, cur_pos, full_loc_arr, cur_direction):
    direction = cur_direction[pid]
    if direction == 'r':
        rotation_size = -90
        eye_pos = np.array([[1, -1], [1, 0]])
    elif direction == 'l':
        rotation_size = 90
        eye_pos = np.array([[0, 0], [0, 1]])
    elif direction == 'u':
        rotation_size = 180
        eye_pos = np.array([[0, -1], [0, 0]])
    elif direction == 'd':
        rotation_size = 0
        eye_pos = np.array([[1, 0], [1, 1]])
    else:
        raise ValueError
    board_state[cur_pos[pid]] = .9
    rotated = np.round(rotate(board_state, rotation_size), decimals=1)
    r_cur_pos = np.nonzero(rotated == .9)
    r_loc_arr = full_loc_arr
    l_center = (int(r_cur_pos[0][0]) + eye_pos[0, 0], int(r_cur_pos[1][0]) + eye_pos[0, 1])
    r_center = (int(r_cur_pos[0][0]) + eye_pos[1, 0], int(r_cur_pos[1][0]) + eye_pos[1, 1])
    center_arr = np.array([list(l_center), list(r_center)])
    to_consider = r_loc_arr[max(l_center[0] + 1, r_center[0] + 1):, :, :]
    is_solid = rotated[max(l_center[0] + 1, r_center[0] + 1):, :] > 0.1
    solid_coords = to_consider[is_solid].astype(float)
    dists = cdist(solid_coords, center_arr + np.array([[1, 0], [1, 0]]))
    dists[dists == 0] = .001
    components = np.tile(solid_coords, (2, 1, 1)) - center_arr[:, None, :]
    components[components == 0] = .001
    val = components[:, :, 1] / components[:, :, 0]
    angles = np.round((np.degrees(np.arctan(val)) - 90) * -32)
    sort_idx = np.argsort(dists, axis=0)
    angles[0] = angles[0, sort_idx[:, 0]]
    angles[1] = angles[1, sort_idx[:, 1]]
    dists = dists.T
    dists[0] = dists[0, sort_idx[:, 0]]
    dists[1] = dists[1, sort_idx[:, 1]]
    sensor = np.zeros((2, 5760))
    visual_field = np.degrees(np.arctan(.5 / dists)) * 32
    max_dist = max(dists.flatten())
    for j in range(2):
        for i in range(len(angles[j])):
            l_bound = max(angles[j, i] - visual_field[j, i], 0)
            r_bound = min(angles[j, i] + visual_field[j, i], sensor.shape[1])
            lum_mod = min(r_bound - l_bound, 1) * (1 - (dists[j, i] / max_dist))
            l_bound = int(np.ceil(l_bound))
            r_bound = int(np.ceil(r_bound))
            mask = np.zeros(r_bound - l_bound)
            mask[sensor[j, l_bound:r_bound] == 0] = 1
            obs_idx = solid_coords[sort_idx[i, j]].astype(int)
            luminance = rotated[obs_idx[0], obs_idx[1]]
            sensor[j, l_bound:r_bound] += luminance * mask * lum_mod
    sensor = zoom(sensor, (1, 1 / 45), order=0)
    return sensor


def partial_observability_filter(global_state, observe_dist, origin, pad_value=.5):
    """
    return the environment state surrounding the current agent position

    :return:
    """
    loc = origin[0]
    if loc[0] < 0:
        raise ValueError("y coord is negative")
    elif loc[1] < 0:
        raise ValueError("x coord is negative")
    elif loc[0] >= global_state.shape[0] - 1:
        raise ValueError("y coord to large")
    elif loc[1] >= global_state.shape[1] - 1:
        print(np.array(global_state))
        raise ValueError("x coord to large")
    up = loc[0]
    lo = loc[0] + 2 * observe_dist + 1
    le = loc[1]
    ri = loc[1] + 2 * observe_dist + 1

    padded = torch.nn.functional.pad(torch.from_numpy(global_state.astype(np.float32)),
                                     (observe_dist, observe_dist, observe_dist, observe_dist),
                                     value=pad_value)

    k_obs = padded[up:lo, le:ri]
    if k_obs.shape != (2 * observe_dist + 1, 2 * observe_dist + 1):
        raise ValueError("Observation dim somehow incorrect")
    return k_obs


def lidar(self, pid):
    """
    gives the distance to each obstacle in a full circle around the agent
    :param pid: agent ID
    :return:
    """
    raise NotImplementedError


    # def observable_env(self, pid=0, mode='basic', vis_res=32):
    #     """
    #     Depends on sensory mode. default can see luminance of vector of 10 pixels across in front
    #     :param pid:
    #     :param vis_res: the visual resolution. Must be a power of 2
    #     :return:
    #     """
    #     if mode == 'vision':
    #         return vision_sim(pid)
    #     elif mode == 'basic':
    #         return bound_box(pid, 5)