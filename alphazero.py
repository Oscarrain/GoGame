from env import *
from env.base_env import *
from torch.nn import Module
from model.linear_model import NumpyLinearModel
from model.linear_model_trainer import NumpyLinearModelTrainer, NumpyModelTrainingConfig
from mcts import puct_mcts

import numpy as np
import random
import torch
import copy
from tqdm import tqdm
from random import shuffle
from players import PUCTPlayer, RandomPlayer, AlphaBetaPlayer

from pit_puct_mcts import multi_match

import logging
logger = logging.getLogger(__name__)


class AlphaZeroConfig():
    def __init__(
        self, 
        n_train_iter:int=300,
        n_match_train:int=20,
        n_match_update:int=20,
        n_match_eval:int=20,
        max_queue_length:int=8000,
        update_threshold:float=0.501,
        n_search:int=200, 
        temperature:float=1.0, 
        C:float=1.0,
        checkpoint_path:str="checkpoint"
    ):
        self.n_train_iter = n_train_iter
        self.n_match_train = n_match_train
        self.max_queue_length = max_queue_length
        self.n_match_update = n_match_update
        self.n_match_eval = n_match_eval
        self.update_threshold = update_threshold
        self.n_search = n_search
        self.temperature = temperature
        self.C = C
        
        self.checkpoint_path = checkpoint_path

class AlphaZero:
    def __init__(self, env:BaseGame, net:NumpyLinearModelTrainer, config:AlphaZeroConfig):
        self.env = env
        self.net = net
        self.last_net = net.copy()
        self.config = config
        self.mcts_config = puct_mcts.MCTSConfig(
            C=config.C, 
            n_search=config.n_search, 
            temperature=config.temperature
        )
        self.mcts_config.with_noise = False
        self.mcts = None
        self.train_eamples_queue = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
    
    def execute_episode(self):
        train_examples = []
        env = self.env.fork()
        state = env.reset()
        config = copy.copy(self.mcts_config)
        config.with_noise = True
        mcts = puct_mcts.PUCTMCTS(env, self.net, config)
        
        while True:
            player = env.current_player
            # MCTS self-play
            ########################
            # TODO: your code here #
            # 使用MCTS计算策略
            policy = mcts.search()
            
            # 获取当前状态的规范形式（从黑方视角）
            canonical_state = env.compute_canonical_form_obs(state, player)
            canonical_state = canonical_state.reshape(-1)  # 展平状态
            
            # 将当前状态和策略添加到训练样本中
            train_examples.append([canonical_state, policy, None])
            
            # 根据策略选择动作
            # 在训练时使用温度参数来增加探索
            if self.config.temperature == 0:
                action = np.argmax(policy)
            else:
                # 将策略转换为概率分布
                policy = policy ** (1/self.config.temperature)
                policy = policy / np.sum(policy)
                action = np.random.choice(len(policy), p=policy)
            
            # 执行动作
            state, reward, done = env.step(action)
            
            if done:
                # 游戏结束时，更新所有样本的价值
                # 从黑方视角计算价值
                value = reward if player == 1 else -reward
                for example in train_examples:
                    example[2] = value
                    value = -value  # 交替更新价值，因为每一步都是从对应玩家的视角
                return train_examples
            
            # 更新MCTS树
            next_mcts = mcts.get_subtree(action)
            if next_mcts is None:
                next_mcts = puct_mcts.PUCTMCTS(env, self.net, config)
            mcts = next_mcts
            ########################
    
    def evaluate(self):
        player = PUCTPlayer(self.mcts_config, self.net, deterministic=True)
        # baseline_player = AlphaBetaPlayer()
        baseline_player = RandomPlayer()
        result = multi_match(self.env, player, baseline_player, self.config.n_match_eval)
        logger.info(f"[EVALUATION RESULT]: win{result[0][0]}, lose{result[0][1]}, draw{result[0][2]}")
        logger.info(f"[EVALUATION RESULT]:(first)  win{result[1][0]}, lose{result[1][1]}, draw{result[1][2]}")
        logger.info(f"[EVALUATION RESULT]:(second) win{result[2][0]}, lose{result[2][1]}, draw{result[2][2]}")
    
    def learn(self):
        for iter in range(1, self.config.n_train_iter + 1):
            logger.info(f"------ Start Self-Play Iteration {iter} ------")
            
            # collect new examples
            T = tqdm(range(self.config.n_match_train), desc="Self Play")
            cnt = ResultCounter()
            for _ in T:
                episode = self.execute_episode()
                self.train_eamples_queue += episode
                cnt.add(episode[0][-1], 1)
            logger.info(f"[NEW TRAIN DATA COLLECTED]: {str(cnt)}")
            
            # pop old examples
            if len(self.train_eamples_queue) > self.config.max_queue_length:
                self.train_eamples_queue = self.train_eamples_queue[-self.config.max_queue_length:]
            
            # shuffle examples for training
            train_data = copy.copy(self.train_eamples_queue)
            shuffle(train_data)
            logger.info(f"[TRAIN DATA SIZE]: {len(train_data)}")
            
            # save current net to last_net
            self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')
            self.last_net.load_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')
            
            # train current net
            self.net.train(train_data)
            
            # evaluate current net
            env = self.env.fork()
            env.reset()
            
            last_mcts_player = PUCTPlayer(self.mcts_config, self.last_net, deterministic=True)
            current_mcts_player = PUCTPlayer(self.mcts_config, self.net, deterministic=True)
            
            result = multi_match(self.env, last_mcts_player, current_mcts_player, self.config.n_match_update)[0]
            # win_rate = result[1] / sum(result)
            total_win_lose = result[0] + result[1]
            win_rate = result[1] / total_win_lose if total_win_lose > 0 else 1
            logger.info(f"[EVALUATION RESULT]: currrent_win{result[1]}, last_win{result[0]}, draw{result[2]}; win_rate={win_rate:.3f}")
            
            if win_rate > self.config.update_threshold:
                self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='best.pth.tar')
                logger.info(f"[ACCEPT NEW MODEL]")
                self.evaluate()
            else:
                self.net.load_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')
                logger.info(f"[REJECT NEW MODEL]")


if __name__ == "__main__":
    from env import *
    import torch
    import matplotlib.pyplot as plt
    
    MASTER_SEED = 0
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    torch.manual_seed(MASTER_SEED)
    
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler("log.txt")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # 记录训练过程中的胜率和不输率
    win_rates = []
    no_lose_rates = []
    
    # Linear
    config = AlphaZeroConfig(
        n_train_iter=100,        # 训练迭代次数
        n_match_train=50,        # 每次迭代的自我对弈次数
        n_match_update=20,
        n_match_eval=20,
        max_queue_length=160000,
        update_threshold=0.45,    # 降低更新阈值，更容易接受新模型
        n_search=400,            # MCTS搜索次数
        temperature=1.5,         # 增加温度，鼓励更多探索
        C=5.0,                   # 增加探索系数
        checkpoint_path="checkpoint/linear_tictactoe"
    )
    
    model_training_config = NumpyModelTrainingConfig(
        epochs=30,               # 增加训练轮数
        batch_size=512,          # 增加批量大小
        lr=0.001,               # 保持较大的学习率
        weight_decay=0.0001      # 保持较小的权重衰减
    )
    
    assert config.n_match_update % 2 == 0
    assert config.n_match_eval % 2 == 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 使用围棋进行调试
    env = GoGame(7, obs_mode="extra_feature")
    
    def base_function(X: np.ndarray) -> np.ndarray:
        return X
    
    net = NumpyLinearModel(env.observation_size, env.action_space_size, None, device=device)
    net = NumpyLinearModelTrainer(env.observation_size, env.action_space_size, net, model_training_config)
    
    # 创建AlphaZero实例
    alphazero = AlphaZero(env, net, config)
    
    # 重写evaluate方法以记录胜率
    original_evaluate = alphazero.evaluate
    def evaluate_with_record():
        player = PUCTPlayer(alphazero.mcts_config, alphazero.net, deterministic=True)
        baseline_player = RandomPlayer()
        result = multi_match(alphazero.env, player, baseline_player, alphazero.config.n_match_eval)
        total = sum(result[0])
        win_rate = result[0][0] / total
        no_lose_rate = (result[0][0] + result[0][2]) / total
        win_rates.append(win_rate)
        no_lose_rates.append(no_lose_rate)
        original_evaluate()
    
    alphazero.evaluate = evaluate_with_record
    
    # 开始训练
    alphazero.learn()
    
    # 绘制胜率和不输率的变化
    plt.figure(figsize=(10, 5))
    iterations = range(len(win_rates))
    
    plt.plot(iterations, win_rates, 'b-', label='Win Rate')
    plt.plot(iterations, no_lose_rates, 'r-', label='No Lose Rate')
    
    plt.xlabel('Iteration')
    plt.ylabel('Rate')
    plt.title('Training Progress against Random Player')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('training_progress.png')
    plt.close()