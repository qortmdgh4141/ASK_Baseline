import os
import sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from absl import app, flags
from src import d4rl_utils
import d4rl
import numpy as np
import tqdm
from dm_control.mujoco import engine


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'kitchen-mixed-v0', '')
flags.DEFINE_integer('cores', '1', '')
flags.DEFINE_integer('iter', '1', '')


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img


def main(_):
    if 'kitchen' in FLAGS.env_name:
        env = d4rl_utils.make_env(FLAGS.env_name)
        dataset = d4rl.qlearning_dataset(env)

        env.reset()

        pixel_dataset = dataset.copy()
        pixel_obs = []
        pixel_next_obs = []
        
        total_samples = len(dataset['observations'])
        cores = FLAGS.cores
        iters = FLAGS.iter
        partition_size = total_samples // cores
        print(f'{cores=}, {iters=}, {partition_size=}')
        target_data = np.arange((iters-1) * partition_size , iters*partition_size)
        if iters==cores:
            target_data = np.arange((iters-1) * partition_size , total_samples)
            
        print(target_data[0], target_data[-1])
        for i in tqdm.tqdm(target_data):
            ob = dataset['observations'][i]
            next_ob = dataset['next_observations'][i]

            env.sim.set_state(np.concatenate([ob[:30], np.zeros(29)]))
            env.sim.forward()
            pixel_ob = kitchen_render(env, wh=64)

            env.sim.set_state(np.concatenate([next_ob[:30], np.zeros(29)]))
            env.sim.forward()
            pixel_next_ob = kitchen_render(env, wh=64)

            pixel_obs.append(pixel_ob)
            pixel_next_obs.append(pixel_next_ob)
        pixel_dataset['observations'] = np.array(pixel_obs)
        pixel_dataset['next_observations'] = np.array(pixel_next_obs)

        print(f'data/d4rl_kitchen_rendered_kitchen-mixed-v0_{iters}.npz')
        np.savez_compressed(f'/home/spectrum/study/ASK_Baseline/data/d4rl_kitchen_rendered_kitchen-mixed-v0_{iters}_all.npz', **pixel_dataset)


if __name__ == '__main__':
    app.run(main)