from typing import Iterable, Optional

import numpy as np
import jax.numpy as jnp
import wandb

Array = jnp.ndarray


def interp2d(
    x: Array,
    y: Array,
    xp: Array,
    yp: Array,
    zp: Array,
    fill_value: Optional[Array] = None,
) -> Array:
    """
    Adopted from https://github.com/adam-coogan/jaxinterp2d

    Bilinear interpolation on a grid. ``CartesianGrid`` is much faster if the data
    lies on a regular grid.
    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`
    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.
    """
    if xp.ndim != 1 or yp.ndim != 1:
        raise ValueError("xp and yp must be 1D arrays")
    if zp.shape != (xp.shape + yp.shape):
        raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    xp = jnp.asarray(xp)
    yp = jnp.asarray(yp)
    zp = jnp.asarray(zp)

    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

    # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    if fill_value is not None:
        oob = jnp.logical_or(
            x < xp[0], jnp.logical_or(x > xp[-1], jnp.logical_or(y < yp[0], y > yp[-1]))
        )
        z = jnp.where(oob, fill_value, z)

    return z


def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None, ]

    _, t, c, h, w = v.shape

    if v.dtype == np.uint8:
        v = np.float32(v) / 255.

    if n_cols is None:
        if v.shape[0] <= 4:
            n_cols = 2
        elif v.shape[0] <= 9:
            n_cols = 3
        else:
            n_cols = 6
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate(
            (v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))

    return v


def save_video(label, step, tensor, fps=15, n_cols=None):
    def _to_uint8(t):
        # If user passes in uint8, then we don't need to rescale by 255
        if t.dtype != np.uint8:
            t = (t * 255.0).astype(np.uint8)
        return t
    if tensor.dtype in [object]:
        tensor = [_to_uint8(prepare_video(t, n_cols)) for t in tensor]
    else:
        tensor = prepare_video(tensor, n_cols)
        tensor = _to_uint8(tensor)

    # Encode sequence of images into gif string
    # clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    # plot_path = (pathlib.Path(logger.get_snapshot_dir())
    #              / 'plots'
    #              / f'{label}_{step}.mp4')
    # plot_path.parent.mkdir(parents=True, exist_ok=True)
    #
    # clip.write_videofile(str(plot_path), audio=False, verbose=False, logger=None)


    # tensor: (t, h, w, c)
    tensor = tensor.transpose(0, 3, 1, 2)
    return wandb.Video(tensor, fps=15, format='mp4')
    # logger.record_video(label, str(plot_path))


def record_video(label, step, renders=None, n_cols=None, skip_frames=1):
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        renders[i] = np.concatenate([render, np.zeros((max_length - render.shape[0], *render.shape[1:]), dtype=render.dtype)], axis=0)
        renders[i] = renders[i][::skip_frames]
    renders = np.array(renders)
    return save_video(label, step, renders, n_cols=n_cols)

def plot_value_map(agent, base_observation, goal_info, i, g_start_time, pretrain_batch, obs, transition_index=None, trajs=None, key_node=None):
    if trajs is not None:
        subgoals = []
        for j, t in enumerate(trajs):
            subgoals.extend(t['cur_obs_subgoal'])
        subgoals = np.array(subgoals).reshape(-1,29)
    
    
    import matplotlib.pyplot as plt
    joint = np.tile(base_observation[2:], (59,46,1))
    goal_infos = np.tile(goal_info, (59,46,1))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.8)
    cmap = plt.cm.bwr
    
    x = np.arange(59)
    y = np.arange(46)
    xx, yy = np.meshgrid(y, x)
    coordinates = np.dstack((yy,xx))
    observations = np.concatenate((coordinates, joint), axis=2)
    value = agent.network(observations, goal_infos, method='hilp_value')[0].transpose(1,0)
    
    # value map
    sc1 = axes[0,0].imshow(value, cmap='Blues_r', interpolation='nearest')
    axes[0,0].set_title('all map value')
    axes[0,0].invert_yaxis() 
    
    # real obs value map
    if key_node is not None:
        key_node_index = np.random.choice(key_node.shape[0], size=5)
        key_x, key_y = pretrain_batch['observations'][key_node_index,0], pretrain_batch['observations'][key_node_index,1]
        sc2 = axes[0,1].scatter(key_x, key_y)
        
        _, I = key_node.kmeans.index.search(x=pretrain_batch['rep_observations'][key_node_index], k=5)
        labels = I[:, :5] 
        for i in range(5):
            x, y = pretrain_batch['observations'][labels[:,i],0], pretrain_batch['observations'][labels[:,i],1]
            sc2 = axes[0,1].scatter(x, y)
        
        # batch_size, obs_dim = pretrain_batch['key_node'].shape
        # real_obs_value = agent.network(pretrain_batch['observations'], np.tile(goal_info, (batch_size,1)), method='hilp_value')[0]
        axes[0,1].set_title('nearest key node')

    # nearest key node map
    batch_size, obs_dim = pretrain_batch['observations'].shape
    real_obs_value = agent.network(pretrain_batch['observations'], np.tile(goal_info, (batch_size,1)), method='hilp_value')[0]
    x, y = pretrain_batch['observations'][:,0], pretrain_batch['observations'][:,1]
    sc2 = axes[0,1].scatter(x, y, c=real_obs_value, cmap=cmap, vmin=-101, vmax=0)
    axes[0,1].set_title('real obs value')
    
    cbar_ax_value = fig.add_subplot(gs[0,2])
    cbar = plt.colorbar(sc2,  cax=cbar_ax_value, label='value')
    # sub goals value map
    
    if trajs is not None and subgoals.shape[0]>batch_size:
        index = np.random.choice(subgoals.shape[0], size=batch_size)
        subgoals = subgoals[index]
        
        subgoals_value = agent.network(subgoals, np.tile(goal_info, (subgoals.shape[0],1)), method='hilp_value')[0]
        x, y = subgoals[:,0], subgoals[:,1]
        sc3 = axes[1,0].scatter(x, y, c=subgoals_value, cmap=cmap, vmin=-101, vmax=0)
        axes[1,0].set_title('subgoals value')
        
        # sub goals identity map
        subgoals_identity = agent.network(subgoals, method='identify')
        sc4 = axes[1,1].scatter(x, y, c=subgoals_identity, cmap=cmap, vmin=0, vmax=1)
        axes[1,1].set_title('subgoals identity')
    
    if 'networks_hilp_value' in agent.network.params.keys():
        # index = np.random.choice(obs.shape[0], size=1024)
        if transition_index is not None:
            filtered_transition_index, hlip_filtered_index, dones_indexes = transition_index
            random_obs_sub = obs[np.random.choice(obs[filtered_transition_index].shape[0], size=filtered_transition_index.sum())]
            hilp_filtered_obs_sub = obs[filtered_transition_index]
            s=5
        
        else:
            filtered_transition_index = transition_index = np.random.choice(obs.shape[0], size=batch_size)
            
            random_obs_sub = hilp_filtered_obs_sub = obs[np.random.choice(obs.shape[0], size=batch_size)]
            s=5
        
        hilp_value = agent.network(hilp_filtered_obs_sub, np.tile(goal_info, (hilp_filtered_obs_sub.shape[0],1)), method='hilp_value')[0]
        obs_value = agent.network(random_obs_sub, np.tile(goal_info, (random_obs_sub.shape[0],1)), method='hilp_value')[0]
        
        x_, y_ = random_obs_sub[:,0], random_obs_sub[:,1]
        x, y = hilp_filtered_obs_sub[:,0], hilp_filtered_obs_sub[:,1]
        
        sc3 = axes[1,0].scatter(x_, y_, c=obs_value, cmap=cmap, s=s)
        axes[1,0].set_title('obs_value random sampled')
        
        # sub goals identity map
        sc4 = axes[1,1].scatter(x, y, c=hilp_value, cmap=cmap, s=s)
        axes[1,1].set_title('obs_hilp_value filtered')
    
    
    cbar_ax_identity = fig.add_subplot(gs[1,2])
    cbar = plt.colorbar(sc4,  cax=cbar_ax_identity, label='probs')
    # plt.gca().invert_yaxis()
    
    import os
    
    dir_name = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(dir_name, 'value_img', g_start_time)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'sampled_obs_img_{i}.png'), format="PNG", dpi=300)
    # plt.close()
    
    # os.makedirs(f'/home/qortmdgh4141/disk/HIQL_Team_Project/TG/value_img/{g_start_time}', exist_ok=True)
    # plt.savefig(f'/home/qortmdgh4141/disk/HIQL_Team_Project/TG/value_img/{g_start_time}/value_img_{i}.png', format="PNG", dpi=300)
    
    
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Read the buffer into a PIL image and convert to NumPy array
    from PIL import Image
    value_map = Image.open(buf)
    value_map = np.array(value_map)
    
    
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.8)
    cmap = plt.cm.bwr
    
    random_index = np.random.choice(obs.shape[0], size=len(filtered_transition_index))
    x, y = obs[random_index,0], obs[random_index,1]
    sc1 = axes[0].scatter(x, y)
    
    x_, y_ = obs[filtered_transition_index, 0], obs[filtered_transition_index, 1]
    sc2 = axes[1].scatter(x_, y_)
    
    axes[0].set_title('all map value')
    # cbar_ax_identity = fig.add_subplot(gs[0,2])
    # cbar = plt.colorbar(sc2,  cax=cbar_ax_identity, label='probs')
    # plt.gca().invert_yaxis()
    
    # os.makedirs(f'/home/qortmdgh4141/disk/HIQL_Team_Project/TG/value_img/{g_start_time}', exist_ok=True)
    # plt.savefig(f'/home/qortmdgh4141/disk/HIQL_Team_Project/TG/value_img/{g_start_time}/sampled_obs_img_{i}.png', format="PNG", dpi=300)
    plt.savefig(os.path.join(save_path, f'identity_img_{i}.png'), format="PNG", dpi=300)
    plt.close()
    
    
    
    
    
    identity_map = None
    if 'networks_identify' in agent.network.params.keys():
        import jax
        # real obs
        identity_map_real_obs = agent.network(pretrain_batch['observations'], method='identify')
        
        x, y = pretrain_batch['observations'][:,0], pretrain_batch['observations'][:,1]
        
        # pseudo obs
        import time
        pseudo_obs = jax.random.uniform(jax.random.PRNGKey(int(time.time())), (3096, base_observation.shape[-1]), minval=pretrain_batch['observations'].min(axis=0), maxval=pretrain_batch['observations'].max(axis=0))
        
        x_, y_ = pseudo_obs[:,0], pseudo_obs[:,1]
        identity_map_pseudo_obs = agent.network(pseudo_obs, method='identify')


        # pseudo obs with real joint
        pseudo_obs_with_dataset_joint = jnp.concatenate([x_.reshape(-1,1), y_.reshape(-1,1), pretrain_batch['observations'][:,2:]], axis=1)
        identity_map_with_real_joint = agent.network(pseudo_obs_with_dataset_joint, method='identify')
        
        # real obs with pseudo joint
        real_obs_with_pseudo_joint = jnp.concatenate([x.reshape(-1,1), y.reshape(-1,1), pseudo_obs[:,2:]], axis=1)
        identity_map_with_pseudo_joint = agent.network(real_obs_with_pseudo_joint, method='identify')
        
        
        
        from matplotlib.gridspec import GridSpec
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.8)
        cmap = plt.cm.bwr
        
        sc1 = axes[0,0].scatter(x, y, c=identity_map_real_obs, cmap=cmap, vmin=0, vmax=1)
        axes[0,0].set_title('real obs')
        
        sc2 = axes[0,1].scatter(x_, y_, c=identity_map_pseudo_obs, cmap=cmap, vmin=0, vmax=1)
        axes[0,1].set_title('pseudo obs 2')
        

        sc3 = axes[1,1].scatter(x, y, c=identity_map_with_pseudo_joint, cmap=cmap, vmin=0, vmax=1)
        axes[1,1].set_title('real obs, pseudo joint 3')
        
        sc4 = axes[1,0].scatter(x_, y_, c=identity_map_with_real_joint, cmap=cmap, vmin=0, vmax=1)
        axes[1,0].set_title('pseudo xy, real joint 4')
        
        cbar_ax = fig.add_subplot(gs[0,2])
        cbar = plt.colorbar(sc4,  cax=cbar_ax, label='probs')
        
        plt.savefig(os.path.join(save_path, 'identity_img_{i}.png'), format="PNG", dpi=300)
        plt.close()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
    
        identity_map = Image.open(buf)
        identity_map = np.array(identity_map)
    
    return value_map, identity_map

def plot_q_map(agent, base_observation, goal_info, i, g_start_time, pretrain_batch, transition_index=None, trajs=None):
    if trajs is not None:
        subgoals = []
        obs = []
        for j, t in enumerate(trajs):
            subgoals.extend(t['cur_obs_subgoal'])
            obs.extend(t['observation'])
        subgoals = np.array(subgoals).reshape(-1,29)
        obs = np.array(obs).reshape(-1,29)
    
    
    import matplotlib.pyplot as plt
    joint = np.tile(base_observation[2:], (59,46,1))
    goal_infos = np.tile(goal_info, (59,46,1))
    base_observation_tiled = np.tile(base_observation, (59,46,1))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.8)
    cmap = plt.cm.bwr
    
    x = np.arange(59)
    y = np.arange(46)
    xx, yy = np.meshgrid(y, x)
    coordinates = np.dstack((yy,xx))
    observations = np.concatenate((coordinates, joint), axis=2)
    q_value = agent.network(base_observation_tiled, observations, goal_infos, method='high_qf')[0].transpose(1,0)
    
    # value map
    sc1 = axes[0,0].imshow(q_value, cmap='Blues_r', interpolation='nearest')
    axes[0,0].set_title('all map q value')
    axes[0,0].invert_yaxis() 
    
    batch_size, obs_dim = pretrain_batch['observations'].shape
    # real obs value map
    real_obs_value = agent.network(np.tile(base_observation, (batch_size,1)), pretrain_batch['observations'], np.tile(goal_info, (batch_size,1)), method='high_qf')[0]
    x, y = pretrain_batch['observations'][:,0], pretrain_batch['observations'][:,1]
    sc2 = axes[0,1].scatter(x, y, c=real_obs_value, cmap=cmap)
    axes[0,1].set_title('real obs q value')
    
    cbar_ax_value = fig.add_subplot(gs[0,2])
    cbar = plt.colorbar(sc2,  cax=cbar_ax_value, label='q value')
    # sub goals value map
    
    if trajs is not None and subgoals.shape[0]>batch_size:
        index = np.random.choice(subgoals.shape[0], size=batch_size)
        subgoals = subgoals[index]
        
        subgoals_value = agent.network(subgoals, np.tile(goal_info, (subgoals.shape[0],1)), method='high_qf')[0]
        x, y = subgoals[:,0], subgoals[:,1]
        sc3 = axes[1,0].scatter(x, y, c=subgoals_value, cmap=cmap)
        axes[1,0].set_title('subgoals q value')
        
        # sub goals identity map
        # subgoals_identity = agent.network(subgoals, method='identify')
        # sc4 = axes[1,1].scatter(x, y, c=subgoals_identity, cmap=cmap, vmin=0, vmax=1)
        # axes[1,1].set_title('subgoals identity')
    
    # if 'networks_hilp_value' in agent.network.params.keys():
        # index = np.random.choice(obs.shape[0], size=1024)
    # if transition_index is not None:
    #     filtered_transition_index, hlip_filtered_index, dones_indexes = transition_index
    #     random_obs_sub = obs[np.random.choice(obs[filtered_transition_index].shape[0], size=filtered_transition_index.sum())]
    #     hilp_filtered_obs_sub = obs[filtered_transition_index]
    #     s=0.1
    
    # else:
    #     filtered_transition_index = transition_index = np.random.choice(obs.shape[0], size=1024)
        
    #     random_obs_sub = hilp_filtered_obs_sub = obs[np.random.choice(obs.shape[0], size=1024)]
    #     s=1
    
    # hilp_value = agent.network(hilp_filtered_obs_sub, np.tile(goal_info, (hilp_filtered_obs_sub.shape[0],1)), method='hilp_value')[0]
    
    import jax 
    
    obs_min = jnp.min(pretrain_batch['observations'], axis=0)
    obs_max = jnp.max(pretrain_batch['observations'], axis=0)
    random_obs_sub = jax.random.uniform(key=agent.rng, shape=(batch_size, pretrain_batch['observations'].shape[-1]), minval=obs_min, maxval=obs_max)
    
    
    
    obs_value = agent.network(np.tile(base_observation, (batch_size,1)), random_obs_sub, np.tile(goal_info, (random_obs_sub.shape[0],1)), method='high_qf')[0]
    
    x_, y_ = random_obs_sub[:,0], random_obs_sub[:,1]
    key_x, key_y = pretrain_batch['key_node'][:,0], pretrain_batch['key_node'][:,1]
    random_obs_value = agent.network(np.tile(base_observation, (batch_size,1)), pretrain_batch['key_node'], np.tile(goal_info, (random_obs_sub.shape[0],1)), method='high_qf')[0]
    
    sc3 = axes[1,0].scatter(x_, y_, c=obs_value, cmap=cmap)
    axes[1,0].set_title('obs_value random sampled')
    
    # sub goals identity map
    sc4 = axes[1,1].scatter(key_x, key_y, c=random_obs_value, cmap=cmap)
    axes[1,1].set_title('obs_value key node')
    
    
    cbar_ax_identity = fig.add_subplot(gs[1,2])
    cbar = plt.colorbar(sc2,  cax=cbar_ax_identity, label='probs')
    # plt.gca().invert_yaxis()
    
    import os
    
    dir_name = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(dir_name, 'q value_img', g_start_time)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'sampled_obs_img_{i}.png'), format="PNG", dpi=300)
    # plt.close()
    
    # os.makedirs(f'/home/qortmdgh4141/disk/HIQL_Team_Project/TG/value_img/{g_start_time}', exist_ok=True)
    # plt.savefig(f'/home/qortmdgh4141/disk/HIQL_Team_Project/TG/value_img/{g_start_time}/value_img_{i}.png', format="PNG", dpi=300)
    
    
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Read the buffer into a PIL image and convert to NumPy array
    from PIL import Image
    value_map = Image.open(buf)
    value_map = np.array(value_map)
    
    
    
    # fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # # from matplotlib.gridspec import GridSpec
    # gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.8)
    # cmap = plt.cm.bwr
    
    # random_index = np.random.choice(obs.shape[0], size=len(filtered_transition_index))
    # x, y = obs[random_index,0], obs[random_index,1]
    # sc1 = axes[0].scatter(x, y, s=s)
    
    # x_, y_ = obs[filtered_transition_index, 0], obs[filtered_transition_index, 1]
    # sc2 = axes[1].scatter(x_, y_, s=s)
    
    # axes[0].set_title('all map value')

    # plt.savefig(os.path.join(save_path, f'identity_img_{i}.png'), format="PNG", dpi=300)
    # plt.close()
    
    
    
    
    
    # identity_map = None
    # if 'networks_identify' in agent.network.params.keys():
    #     import jax
    #     # real obs
    #     identity_map_real_obs = agent.network(pretrain_batch['observations'], method='identify')
        
    #     x, y = pretrain_batch['observations'][:,0], pretrain_batch['observations'][:,1]
        
    #     # pseudo obs
    #     import time
    #     pseudo_obs = jax.random.uniform(jax.random.PRNGKey(int(time.time())), (3096, base_observation.shape[-1]), minval=pretrain_batch['observations'].min(axis=0), maxval=pretrain_batch['observations'].max(axis=0))
        
    #     x_, y_ = pseudo_obs[:,0], pseudo_obs[:,1]
    #     identity_map_pseudo_obs = agent.network(pseudo_obs, method='identify')


    #     # pseudo obs with real joint
    #     pseudo_obs_with_dataset_joint = jnp.concatenate([x_.reshape(-1,1), y_.reshape(-1,1), pretrain_batch['observations'][:,2:]], axis=1)
    #     identity_map_with_real_joint = agent.network(pseudo_obs_with_dataset_joint, method='identify')
        
    #     # real obs with pseudo joint
    #     real_obs_with_pseudo_joint = jnp.concatenate([x.reshape(-1,1), y.reshape(-1,1), pseudo_obs[:,2:]], axis=1)
    #     identity_map_with_pseudo_joint = agent.network(real_obs_with_pseudo_joint, method='identify')
        
        
        
    #     from matplotlib.gridspec import GridSpec
    #     fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    #     gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.8)
    #     cmap = plt.cm.bwr
        
    #     sc1 = axes[0,0].scatter(x, y, c=identity_map_real_obs, cmap=cmap, vmin=0, vmax=1)
    #     axes[0,0].set_title('real obs')
        
    #     sc2 = axes[0,1].scatter(x_, y_, c=identity_map_pseudo_obs, cmap=cmap, vmin=0, vmax=1)
    #     axes[0,1].set_title('pseudo obs 2')
        

    #     sc3 = axes[1,1].scatter(x, y, c=identity_map_with_pseudo_joint, cmap=cmap, vmin=0, vmax=1)
    #     axes[1,1].set_title('real obs, pseudo joint 3')
        
    #     sc4 = axes[1,0].scatter(x_, y_, c=identity_map_with_real_joint, cmap=cmap, vmin=0, vmax=1)
    #     axes[1,0].set_title('pseudo xy, real joint 4')
        
    #     cbar_ax = fig.add_subplot(gs[0,2])
    #     cbar = plt.colorbar(sc4,  cax=cbar_ax, label='probs')
        
    #     plt.savefig(os.path.join(save_path, 'identity_img_{i}.png'), format="PNG", dpi=300)
    #     plt.close()
        
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # buf.seek(0)
    
        # identity_map = Image.open(buf)
        # identity_map = np.array(identity_map)
    
    return value_map

class CsvLogger:
    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
