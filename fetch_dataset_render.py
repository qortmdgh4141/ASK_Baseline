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
flags.DEFINE_string('env_name', 'visual-FetchPush-v1-mixed', '')
flags.DEFINE_integer('cores', '1', '')
flags.DEFINE_integer('iter', '1', '')


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img


# def main(_):
#     if 'kitchen' in FLAGS.env_name:
#         env = d4rl_utils.make_env(FLAGS.env_name)
#         dataset = d4rl.qlearning_dataset(env)

#         env.reset()

#         pixel_dataset = dataset.copy()
#         pixel_obs = []
#         pixel_next_obs = []
        
#         total_samples = len(dataset['observations'])
#         cores = FLAGS.cores
#         iters = FLAGS.iter
#         partition_size = total_samples // cores
#         print(f'{cores=}, {iters=}, {partition_size=}')
#         target_data = np.arange((iters-1) * partition_size , iters*partition_size)
#         if iters==cores:
#             target_data = np.arange((iters-1) * partition_size , total_samples)
            
#         print(target_data[0], target_data[-1])
#         for i in tqdm.tqdm(target_data):
#             ob = dataset['observations'][i]
#             next_ob = dataset['next_observations'][i]

#             env.sim.set_state(np.concatenate([ob[:30], np.zeros(29)]))
#             env.sim.forward()
#             pixel_ob = kitchen_render(env, wh=64)

#             env.sim.set_state(np.concatenate([next_ob[:30], np.zeros(29)]))
#             env.sim.forward()
#             pixel_next_ob = kitchen_render(env, wh=64)

#             pixel_obs.append(pixel_ob)
#             pixel_next_obs.append(pixel_next_ob)
#         pixel_dataset['observations'] = np.array(pixel_obs)
#         pixel_dataset['next_observations'] = np.array(pixel_next_obs)

#         print(f'data/d4rl_kitchen_rendered_kitchen-mixed-v0_{iters}.npz')
#         np.savez_compressed(f'/home/spectrum/study/ASK_Baseline/data/d4rl_kitchen_rendered_kitchen-mixed-v0_{iters}_all.npz', **pixel_dataset)

def main(_):
    import pickle
    from src.envs.fetch_visual import fetch_load, FetchPushImage
    kwargs = {'rand_y':True, 'height':64, 'width':64, 'render_mode':'rgb_array'}
    env = FetchPushImage(**kwargs)
    env.reset()
    visual, env_name, version, type_ = FLAGS.env_name.split('-')
    dataset_file = os.path.join(f'/home/spectrum/study/ASK_Baseline/data/{type_}/{env_name}/buffer.pkl')
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
        print(f'{dataset_file}, fetch dataset loaded')
        
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(10,8))
    state = env.sim.get_state()
    for i in range(50):
        state[1][0:3] = dataset['o'][0][i][0:3] # end effet x,y,z
        state[1][15:18] = dataset['o'][0][i][3:6] # puck x,y,z
        # print(state[1][:3], state[1][15:18])
        env.goal = dataset['g'][0][i]
        env.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", dataset['o'][0][i][9]) # gripper left
        env.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", dataset['o'][0][i][10]) # gripper right
        
        # 데이터 셋으로부터 state에 적합한 값을 지정하고, set_state()로 설정 후 rendering할것
        # sim.get_state() 
        env.sim.set_state(state)
        # print(env.sim.data.site_xpos)
        plt.imshow(env.render())
        plt.savefig(f'/home/spectrum/study/ASK_Baseline/img_fetch/img_{i}.png')
    plt.close()
    pass

def main_(_):
    import pickle
    from src.envs.fetch_visual import fetch_load, FetchPushImage
    visual, env_name, version, type_ = FLAGS.env_name.split('-')
    dataset_file = os.path.join(f'/home/spectrum/study/ASK_Baseline/data/{type_}/{env_name}/buffer.pkl')
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
        print(f'{dataset_file}, fetch dataset loaded')
    kwargs = {'rand_y':True, 'height':64, 'width':64, 'render_mode':'rgb_array'}
    
    initial_qpos = {}
    initial_qpos["robot0:slide0"] = dataset['o'][0][0][0]
    initial_qpos["robot0:slide1"] = dataset['o'][0][0][1]
    initial_qpos["robot0:slide2"] = dataset['o'][0][0][2]
    initial_qpos["object0:joint"] = np.concatenate((dataset['o'][0][0][3:6], np.array([1.0, 0.0, 0.0, 0.0])))
    pos = {'initial_qpos':initial_qpos}
    kwargs.update(pos)
    env = FetchPushImage(**kwargs)
    env.reset()


        
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(10,8))
    state = env.sim.get_state()
    # state[1][0:3] = dataset['o'][0][0][0:3] # end effet x,y,z
    k = 1
    state[1][15:18] = dataset['o'][k][0][3:6] # puck x,y,z
    # # print(state[1][:3], state[1][15:18])
    env.goal = dataset['g'][k][0]
    # env.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", dataset['o'][0][0][9]) # gripper left
    # env.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", dataset['o'][0][0][10]) # gripper right
    # env.sim.set_state(state)
    for i in range(49):
        s, r, d, info = env.step(dataset['u'][k][i])
        # 데이터 셋으로부터 state에 적합한 값을 지정하고, set_state()로 설정 후 rendering할것
        # sim.get_state() 
        # print(env.sim.data.site_xpos)
        plt.imshow(s)
        plt.savefig(f'/home/spectrum/study/ASK_Baseline/img_fetch/actions5/img_{i}.png')
    plt.close()
    pass

if __name__ == '__main__':
    app.run(main_)
    
    
