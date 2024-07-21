import argparse
import numpy as np
import faiss
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from tqdm import tqdm
from random import choice


def build_keynodes(dataset, flags=None, episode_index= None, rep_obs=None):
    obs = dataset['observations'][episode_index]
    rep_obs = rep_obs[episode_index]
    
    # if flags.specific_dim_On:
    #     if 'ant' in flags.env_name:
    #         ant_episode_length = 1000
    #         obs = obs[:, :2]
    #     elif 'kitchen' in flags.env_name:
    #         obs = obs[:, :9]
    #     elif 'calvin' in flags.env_name:
    #         obs = obs[:, :15]
    
    data_index = np.arange(len(obs))
    # if flags.sparse_data:
    #     assert int(flags.sparse_data) in [-7, -9, 0] # 'For small data setting, choose from the following : [-7, -9, 0]. [30%, 10%, 100%], respectively'
    #     if flags.sparse_data < 0:
    #         if 'ant' in flags.env_name:
    #             data_count = int((10+flags.sparse_data)*0.1 * len(dataset['observations']) / ant_episode_length)
    #             episode_count = episode_index.shape[0]
    #             sparse_data_episode_index = np.random.choice(episode_count, data_count)
    #             sparse_data_index = episode_index[sparse_data_episode_index].reshape(-1)
    #         elif 'kitchen' in flags.env_name: # 질문 승호코드체크
    #             episode_count = len(episode_index)
    #             sparse_count = int((10+flags.sparse_data)*0.1 * episode_count)
    #             sparse_data_episode_index = np.random.choice(episode_count, sparse_count)
    #             sparse_data_index = np.hstack(episode_index[sparse_data_episode_index]).astype(np.int)
    #         value_index = dataset['returns'][sparse_data_index]>0
    #         non_value_index = dataset['returns'][sparse_data_index]==0            
    #         data_index=sparse_data_index 
    #         print(f'data : {len(sparse_data_index)}, non expert data : {non_value_index.sum()}, expert_data : {value_index.sum()}, ratio :  {np.round(value_index.sum() / len(sparse_data_index) * 100, 2)} %')
    # else:
        # data_index = episode_index
    
    # if flags.kmean_weight_On:
        # nodes = KeyNode(obs=obs[data_index], values=dataset['returns'][data_index], flags=flags)
    # else:
        # state_values = np.ones(obs.shape[0]).astype(np.float32) 
    nodes = KeyNode(obs=obs, rep_obs=rep_obs, flags=flags)
            
    nodes.construct_nodes(spherical_On=flags.spherical_On) 
    nodes.visualize_key_nodes(flags)
    
    return nodes, data_index

def kmeans_pp_compute_distances(points, clusters):
    def distance(p, c):
        return jnp.linalg.norm(p - c)
    def min_distance(point):
        return jnp.min(jax.vmap(lambda c: distance(point, c))(clusters))
    return jax.vmap(min_distance)(points)
kmeans_pp_compute_distances_jit = jax.jit(kmeans_pp_compute_distances)


class KeyNode(object):
    def __init__(self,
                 obs: np.ndarray,
                 rep_obs : np.ndarray,
                #  values: np.ndarray,
                 flags : str = None):

        self.flags = flags
        self.env_name = flags.env_name
        self.keynode_num = flags.keynode_num
        self.obs = jnp.array(obs)
        self.rep_obs = rep_obs  
        # self.f_s = np.array(obs) 
        # self.values = values 
        self.spherical_On = bool(flags.spherical_On)
        if self.flags.specific_dim_On:
            if 'ant' in self.env_name:
                node_dim = 2
            elif 'kitchen' in self.env_name:
                node_dim = 9
            elif 'calvin' in self.env_name:
                node_dim = 15
        self.node_dim = node_dim
        self.reduced_f_s = None  
        self.weighted_values = None
        self.labels = None  
        self.graph = None  
        
        self.kmean_weight_On = flags.kmean_weight_On

        self.find_nodes = jax.vmap(self.find_node_pos)
        self.find_node = self.find_node_pos
        
        self.find_node_in_datasets = jax.vmap(self.find_node_pos_in_distance)
        self.find_node_in_dataset = self.find_node_pos_in_distance
        
        # self.find_nodes = jax.jit(jax.vmap(self.find_node_pos))
        # self.find_node = jax.jit(self.find_node_pos)        

    def construct_nodes(self, rep_observations=None, spherical_On=0.0):
        if rep_observations is None:
            # 0610 승호수정 spherical
            # self.reduced_f_s, self.weighted_values, self.labels = self.sparse_node(f_s=self.f_s, values=self.values, keynode_num=self.keynode_num, spherical_On=spherical_On)
            self.centroids, self.rep_centroids, self.indexes = self.sparse_node(f_s=self.rep_obs, keynode_num=self.keynode_num, spherical_On=spherical_On)
            self.graph = self.create_nodes(reduced_f_s=self.centroids)
            # self.graph = self.create_nodes(reduced_f_s=self.centroids, weighted_values=self.weighted_values)
        else:
            self.rep_f_s = np.array(rep_observations) 
            # 0610 승호수정 spherical
            self.rep_reduced_f_s, self.rep_weighted_values, self.rep_labels = self.sparse_node(f_s=self.rep_f_s, values=self.values, keynode_num = self.keynode_num, spherical_On=spherical_On)
            self.graph = self.create_nodes(reduced_f_s=self.rep_reduced_f_s, weighted_values=self.weighted_values)
    
    def init_kmeans_pp(self, points):
        clusters = []
        indexes = []

        index = np.random.choice(len(points))
        clusters.append(points[index])  # first centroid is random point
        indexes.append(index)
        # clusters.append(choice(points))  # first centroid is random point
        
        for _ in tqdm(range(self.keynode_num - 1), desc="Initializing centroids"):  # for other centroids
            distances = kmeans_pp_compute_distances_jit(points, jnp.array(clusters))
            index = np.argmax(distances)
            clusters.append(points[index])
            indexes.append(index.copy())
            
        return jnp.array(clusters), jnp.array(indexes)
    
    def sparse_node(self,
                    f_s: np.ndarray,
                    # values: np.ndarray,
                    keynode_num : int,
                    spherical_On : float):
        niter = 1  
        nredo = 10
        verbose = True 
        update_index = True
        d = f_s.shape[1]
        # max_points_per_centroid = self.f_s.shape[0] // self.keynode_num
        # self.keynode_dim = f_s.shape[1]
        

        # 0610 승호수정 spherical
        # self.spherical_On = bool(spherical_On) 
        # if self.spherical_On:
        #     f_s = f_s / np.sqrt(d)
        # else:
        #     f_s_max, f_s_min = np.max(f_s, axis=0),  np.min(f_s, axis=0)
        #     f_s = (f_s - f_s_min) / (f_s_max - f_s_min)
        #     self.scale_min, self.scale_max  = f_s_min, f_s_max
        
        scaled_f_s = f_s
        # Initialize centroids using k-means++
        self.rep_centroids, self.indexes = self.init_kmeans_pp(scaled_f_s)
        self.centroids = self.obs[self.indexes]
        self.normalized_centroids = self.centroids / jnp.linalg.norm(self.centroids, axis=-1, keepdims=True)
        self.normalized_rep_centroids = self.rep_centroids / jnp.linalg.norm(self.rep_centroids, axis=-1, keepdims=True)
        
        # 0610 승호수정 spherical
        # self.kmeans = faiss.Kmeans(d=self.keynode_dim, k=self.keynode_num, seed=self.flags.seed, spherical=self.spherical_On,
        #                       niter=niter, nredo=nredo, max_points_per_centroid=max_points_per_centroid,  update_index=update_index, verbose=verbose, gpu=True)
        
        # if self.kmean_weight_On: 
        #     self.kmeans.train(x=scaled_f_s, weights=self.values, init_centroids=init_centroids)
        # else:
        #     self.kmeans.train(x=scaled_f_s, init_centroids=init_centroids)
        
  
        # reduced_f_s = self.kmeans.centroids
        # self.reduced_pos = reduced_f_s
        
        # 0610 승호수정 spherical
        # if self.spherical_On:
        #     recovered_f_s = reduced_f_s * np.sqrt(d) # reduced_f_s 원래 스케일로 복원
        # else:
        #     recovered_f_s = reduced_f_s * (f_s_max - f_s_min) + f_s_min # reduced_f_s 원래 스케일로 복원
        # self.pos = recovered_f_s
        
        # _, I = self.kmeans.index.search(f_s, 1) 
        # labels = I[:, 0] # I: f_s의 각 요소(노드)에 대해 가장 가까운 centroid의 인덱스
        # weighted_values = self.calculate_weighted_values(labels, values)  
        
        # self.pos = f_s[labels] # 각 노드의 위치를 centroid의 위치로 설정
        
        # print(f"Offline Dataset => Clustered Nodes  /   {len(f_s)} => {len(reduced_f_s)}")
        # return reduced_f_s, weighted_values, labels
        return self.centroids, self.rep_centroids, self.indexes

    def calculate_weighted_values(self,
                                  labels: np.ndarray,
                                  values: np.ndarray):
        unique_labels = np.unique(labels)
        weighted_values = np.zeros(len(labels)) 
        
        # 질문 이 코드대로라면 weighted_values에는 0인값이 대부분채워질텐데? 이렇게 사용하는것을 추구한게 맞나? (딕셔너리로 구현하는게 맞지 않을까?)
        for i, unique_label in enumerate(unique_labels):
            cluster_values_sum = np.sum(values[labels == unique_label])  
            cluster_size = np.sum(labels == unique_label)  
            weighted_values[unique_label] = cluster_values_sum / cluster_size  

        return weighted_values
    
    def create_nodes(self,
                     reduced_f_s: np.ndarray,
                     weighted_values: np.ndarray = None):
        graph = nx.DiGraph()
        for i, pos in enumerate(reduced_f_s):
            # graph.add_node(i, pos=pos, weighted_value=weighted_values[i])  
            graph.add_node(i, pos=pos)  
        nodes = []
        pos = []
        for node, data in graph.nodes(data=True):
            nodes.append(node)
            pos.append(data['pos'])
        self.nodes = jnp.array(nodes)
        # self.pos = jnp.array(pos)
        
        return graph    
    
    def find_node_pos(self, input_obs):
        input_obs = input_obs.reshape(-1, self.flags.hilp_skill_dim)
        hilp_observation, hilp_subgoal = input_obs[0], input_obs[1] 
        
        # temp distance 계산 - observation 과 subgoal 간 거리
        temp_distance = jnp.linalg.norm(hilp_observation - hilp_subgoal).mean()
        # observation과 rep key node간 거리 계산
        # keynode_distance = jnp.linalg.norm(self.rep_centroids - hilp_observation, axis=-1)
        # subgoal 과 rep key node간 거리 계산
        keynode_distance = jnp.linalg.norm(self.rep_centroids - hilp_subgoal, axis=-1)
        # observation과 rep key node간 거리 계산
        keynode_distance_obs = jnp.linalg.norm(self.rep_centroids - hilp_observation, axis=-1)
        ##############################################################################
        # cos similarity 계산
        # 원점 기준 계산
        # normalized_subgoal = hilp_subgoal / jnp.linalg.norm(hilp_subgoal)
        # cosine_similarity = jnp.dot(self.normalized_rep_centroids, normalized_subgoal)
        
        # observations 기준 계산
        # normalized_subgoal = (hilp_subgoal - hilp_observation) / jnp.linalg.norm(hilp_subgoal - hilp_observation)
        # normalized_rep_centroids = (self.rep_centroids - hilp_observation) / jnp.linalg.norm((self.rep_centroids - hilp_observation))
        # cosine_similarity = jnp.dot(normalized_rep_centroids, normalized_subgoal.T)
        
        
        # # temp distance 2배 이내의 key node 후보 탐색
        # max_distance = jax.lax.cond(temp_distance * 1.2 > keynode_distance.min(),
        #                     lambda _: temp_distance * 1.2,
        #                     lambda _: keynode_distance.min() * 1.2,
        #                     operand=None)
        
        # min_distance = jax.lax.cond(temp_distance * 0.5 > keynode_distance.min(),
        #                             lambda _: temp_distance * 0.5,
        #                             lambda _: keynode_distance.min(),
        #                             operand=None)
        
        
        # cosine_similarity = jnp.where(keynode_distance < max_distance, cosine_similarity, 0)
        # cosine_similarity = jnp.where(keynode_distance < min_distance, 0, cosine_similarity)
        # keynode_index = jnp.argmax(cosine_similarity)
        ##############################################################################
        # if cosine_similarity < 0.8:
        #     cur_obs_key_node = None
        #     cur_obs_latent_key_node = hilp_subgoal
        #     return temp_distance, cur_obs_latent_key_node, hilp_subgoal, cur_obs_key_node
        # # euclidean distance
        # distance = jnp.linalg.norm(self.centroids - input_obs, axis=1)
        
        min_distance = jax.lax.cond(temp_distance * 0.5 > keynode_distance_obs.min(),
                            lambda _: temp_distance * 0.5,
                            lambda _: keynode_distance_obs.min(),
                            operand=None)
        # obs 와 너무 가까운 key node는 제거
        keynode_distance = jnp.where(keynode_distance_obs > min_distance, keynode_distance, 1e6)
        
        # subgoal 과 너무 먼 key node는 제거
        max_distance = jax.lax.cond(temp_distance * 0.5 > keynode_distance.min(),
                            lambda _: temp_distance * 0.5,
                            lambda _: keynode_distance.min() * 1.2,
                            operand=None)
        keynode_distance = jnp.where(keynode_distance < max_distance, keynode_distance, 1e6)
        
        keynode_index = jnp.argmin(keynode_distance)
        
        # cosine_similarity = jnp.dot(self.normalized_rep_centroids, normalized_input_obs.T)
        # index = jnp.argmax(cosine_similarity)
        
        # # euclidean distance
        # distance = jnp.linalg.norm(self.rep_centroids - input_obs, axis=1)
        # index = jnp.argmin(distance)
        
        
        
        # 여러개 sampling
        # cur_obs_key_node = self.centroids[keynode_index[1]]
        # selected_subgoal = subgoal[keynode_index[0]]
        # 1개 sampling
        cur_obs_key_node = self.centroids[keynode_index]
        cur_obs_latent_key_node = self.rep_centroids[keynode_index]
        # hilp_subgoal = hilp_subgoal

                            
        return temp_distance, cur_obs_latent_key_node, hilp_subgoal, cur_obs_key_node

    def find_node_pos_in_distance(self, input_obs):
        hilp_observation = input_obs
        
        # temp distance 계산 - observation 과 subgoal 간 거리
        # temp_distance = jnp.linalg.norm(hilp_observation - hilp_subgoal).mean()
        # obs 와 rep key node간 거리 계산
        keynode_distance = jnp.linalg.norm(self.rep_centroids - hilp_observation, axis=-1)
        
        keynode_index = jnp.argmin(keynode_distance)
        cur_obs_key_node = self.centroids[keynode_index]
        cur_obs_latent_key_node = self.rep_centroids[keynode_index]
                            
        return cur_obs_key_node, cur_obs_latent_key_node
    
    
    
    
    def find_closest_node(self, input_obs_batch: jnp.ndarray):
        if len(input_obs_batch.shape) == 1:
            temp_distance, cur_obs_latent_key_node, hilp_subgoal, cur_obs_key_node = self.find_node(input_obs_batch)
        else:
            temp_distance, cur_obs_latent_key_node, hilp_subgoal, cur_obs_key_node = self.find_nodes(input_obs_batch)
        return temp_distance, cur_obs_latent_key_node, hilp_subgoal, cur_obs_key_node #

    def find_key_node_in_dataset(self, input_obs_batch: jnp.ndarray):
        if len(input_obs_batch.shape) == 1:
            cur_obs_key_node = self.find_node_in_dataset(input_obs_batch)
        else:
            cur_obs_key_node = self.find_node_in_datasets(input_obs_batch)
        return cur_obs_key_node # 현재 코드에서는 4번쨰 closest_node_observations만 사용
  
    # def find_centroid(self, input_obs):
    #     input_obs = np.array(input_obs).astype(np.float32)
        
    #     if self.spherical_On:
    #         input_pos = input_obs / np.sqrt(self.node_dim)
    #     else:
    #         input_pos = (input_obs - self.scale_min) / (self.scale_max - self.scale_min) 
        
    #     if self.spherical_On:
    #         cosine_similarity = np.dot(self.reduced_f_s, input_pos[:,:self.node_dim].T)
    #         index = np.argmax(cosine_similarity)
    #         # distance = cosine_similarity

    #     else:
    #         distance = np.linalg.norm(self.pos - input_pos[:,:self.node_dim], axis=1)
    #         index = np.argmin(distance)
        
    #     # _, I = self.kmeans.index.search(input_pos[:,:self.node_dim], 1) 
    #     # index = I[0,0] # I: f_s의 각 요소(노드)에 대해 가장 가까운 centroid의 인덱스
    #     return np.concatenate((self.pos[index], input_obs[0,self.node_dim:]))
    
    def visualize_key_nodes(self, flags, node_colors="value_color", figsize=(27, 21), node_size=500, label_size=2, dpi=300 ):
        import time
        t = time.strftime('%m-%d_%H:%M')
        import os
        import sys
        dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        if not os.path.exists(dir_path+'/graph_img/' + flags.env_name):
            os.makedirs(dir_path +'/graph_img/' + flags.env_name , exist_ok = True)
        
        filename = dir_path
        filename += '/graph_img/'
        filename += flags.env_name + '/_' 
        filename += t + '_' 
        filename += str(self.keynode_num) + '_' 
        if flags.sparse_data:
            filename += 'sparse_' + str(flags.sparse_data) + '_'
        if flags.kmean_weight_On == 0:
            filename += 'no_weight_'
        if self.graph is None:
            print("Graph not initialized.")

        elif 'ant' in flags.env_name:
            
            plt.figure(figsize=figsize)
            pos = nx.get_node_attributes(self.graph, 'pos')

            if node_colors is None:
                node_color = 'skyblue'
                nx.draw(self.graph, with_labels=True, node_color=node_color, font_weight='bold', node_size=node_size,
                        font_size=label_size)
                filename += '.png'
                plt.savefig(filename, format="PNG", dpi=dpi)
            else:
                
                for k,v in pos.items():
                    pos[k] = v[:2]
                
                nx.draw(self.graph, pos, with_labels=True, node_color='red', node_size=node_size, font_size=label_size)
                filename += '.png'
                plt.savefig(filename, format="PNG", dpi=dpi)
        
        elif 'kitchen' in flags.env_name:
            pass
            # import pandas as pd
            
            # fig = plt.figure(figsize=(9, 6))
            # ax = fig.add_subplot(111, projection='3d')
            
            # pos = nx.get_node_attributes(self.graph, 'pos')
            # p = pd.DataFrame([pos[k] for k in pos.keys()])
            
            # # plot on x,y,z 
            # weighted_values = nx.get_node_attributes(self.graph, 'weighted_value')
            # min_val = min(weighted_values)
            # max_val = max(weighted_values)
            # norm_colors = [(value - min_val) / (max_val - min_val) for value in weighted_values]
            # cmap = plt.cm.bwr
            # ax.scatter(p[0], p[1], p[2], c = cmap(norm_colors), s=5)
            # filename_xyz = filename + '_xyz.png'
            # plt.savefig(filename_xyz, format="PNG", dpi=dpi)
            
            # # plot TSNE
            # from sklearn.manifold import TSNE
            # fig = plt.figure(figsize=(9, 6))
            # ax = fig.add_subplot(111, projection='3d')
            
            # tsne = TSNE(n_components = 3, n_jobs=32).fit_transform(p)
            # tsne_down = pd.DataFrame(tsne, columns = ['0', '1', '2'])
            # weighted_values = nx.get_node_attributes(self.graph, 'weighted_value')
            # min_val = min(weighted_values)
            # max_val = max(weighted_values)
            # norm_colors = [(value - min_val) / (max_val - min_val) for value in weighted_values]
            # cmap = plt.get_cmap('plasma')
            # ax.scatter(tsne_down['0'], tsne_down['1'], tsne_down['2'], c = cmap(norm_colors), s=5)
            
            # filename_tsne = filename + '_tsne.png'
            # plt.savefig(filename_tsne, format="PNG", dpi=dpi)
            
                


