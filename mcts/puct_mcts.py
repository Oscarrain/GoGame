from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

from model.linear_model_trainer import NumpyLinearModelTrainer
import numpy as np


class PUCTMCTS:
    def __init__(self, init_env:BaseGame, model: NumpyLinearModelTrainer, config: MCTSConfig, root:MCTSNode=None):
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        env = init_env.fork()
        obs = env.observation
        self.root = MCTSNode(
            action=None, env=env, reward=0
        )
        # compute and save predicted policy
        child_prior, _ = self.model.predict(env.compute_canonical_form_obs(obs, env.current_player))
        self.root.set_prior(child_prior)
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)
        else:
            return None
    
    def puct_action_select(self, node:MCTSNode):
        # select the best action based on PUCB when expanding the tree
        
        ########################
        # TODO: your code here #
        ########################
        # 计算所有动作的PUCB值
        puct_values = np.zeros(node.n_action) - INF  # 初始化为负无穷
        total_visits = np.sum(node.child_N_visit)
        
        # 只为合法动作计算PUCB值
        legal_actions = np.where(node.action_mask)[0]
        for action in legal_actions:
            if node.child_N_visit[action] == 0:
                # 如果动作未被访问过，使用先验概率
                puct_values[action] = node.child_priors[action] * self.config.C * np.sqrt(total_visits + 1)
            else:
                # 计算PUCB值
                Q = node.child_V_total[action] / node.child_N_visit[action]
                puct_values[action] = Q + \
                            self.config.C * node.child_priors[action] * \
                            np.sqrt(total_visits) / (1 + node.child_N_visit[action])
        
        # 选择PUCB值最大的动作
        return np.argmax(puct_values)
        ########################

    def backup(self, node:MCTSNode, value):
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        
        ########################
        # TODO: your code here #
        ########################
        # 从叶子节点回溯到根节点
        current = node
        while current is not None:
            # 如果节点有父节点，更新父节点的子节点统计信息
            if current.parent is not None:
                parent = current.parent
                action = current.action
                # 更新子节点的访问次数和总价值
                parent.child_N_visit[action] += 1
                parent.child_V_total[action] += value
            
            # 移动到父节点
            current = current.parent
        ########################  
    
    def pick_leaf(self):
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        
        ########################
        # TODO: your code here #
        ########################
        node = self.root
        
        # 从根节点开始，选择动作直到到达叶子节点
        while not node.done:
            # 如果节点未被访问过，返回该节点
            total_visits = np.sum(node.child_N_visit)
            if total_visits == 0:
                return node
            
            # 选择动作
            action = self.puct_action_select(node)
            
            # 如果动作对应的子节点不存在，创建新节点
            if not node.has_child(action):
                # 创建新节点
                child = node.add_child(action)
                
                # 如果是新节点，计算并保存先验概率
                if not child.done:
                    obs = child.env.compute_canonical_form_obs(child.env.observation, child.env.current_player)
                    child_prior, _ = self.model.predict(obs)
                    child.set_prior(child_prior)
                
                return child
            
            # 移动到子节点
            node = node.get_child(action)
        
        return node
        ########################
    
    def get_policy(self, node:MCTSNode = None):
        # return the policy of the tree(root) after the search
        # the policy conmes from the visit count of each action 
        
        ########################
        # TODO: your code here #
        ########################
        if node is None:
            node = self.root
        
        # 初始化策略为0
        policy = np.zeros(node.n_action)
        
        # 获取合法动作
        legal_actions = np.where(node.action_mask)[0]
        
        # 计算合法动作的访问次数总和
        total_visits = np.sum(node.child_N_visit[legal_actions])
        
        # 如果没有任何访问，返回均匀分布（仅在合法动作中）
        if total_visits == 0:
            policy[legal_actions] = 1.0 / len(legal_actions)
        else:
            # 根据访问次数计算策略（仅在合法动作中）
            policy[legal_actions] = node.child_N_visit[legal_actions] / total_visits
        
        return policy
        ########################

    def search(self):
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                ########################
                # TODO: your code here #
                ########################
                # 如果游戏结束，使用最终奖励作为价值
                value = leaf.reward
                ########################
            else:
                ########################
                # TODO: your code here #
                ########################
                # 使用价值模型预测价值
                obs = leaf.env.compute_canonical_form_obs(leaf.env.observation, leaf.env.current_player)
                _, value = self.model.predict(obs)
                value = value[0]  # 因为predict返回的是数组
                ########################
            self.backup(leaf, value)
            
        return self.get_policy(self.root)