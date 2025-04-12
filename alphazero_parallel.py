import multiprocessing.connection
from env.base_env import BaseGame, get_symmetries, ResultCounter
from env.go.go_env import GoGame
from torch.nn import Module
from model.linear_model import NumpyLinearModel
from model.linear_model_trainer import NumpyLinearModelTrainer, NumpyModelTrainingConfig
from mcts import puct_mcts

import torch
import multiprocessing, os, time, sys
import dill, pickle
import traceback

import numpy as np
import random
import copy
from tqdm import tqdm
from random import shuffle
from players import PUCTPlayer, RandomPlayer, AlphaBetaPlayer
from alphazero import AlphaZeroConfig

from pit_puct_mcts import multi_match

import logging
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

class AlphaZeroParallelConfig(AlphaZeroConfig):
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

def execute_episode_worker(
    conn:multiprocessing.connection.Connection, 
    env_builder,
    net_builder, 
    mcts_config:puct_mcts.MCTSConfig, 
    id:int, 
    seed:int,
    checkpoint_path:str="",
    ):
    logger.debug(f"[Worker {id}] Initializing worker {id}")
    st0 = time.time()
    env_builder = dill.loads(env_builder)
    net_builder = dill.loads(net_builder)
    root_env:BaseGame = env_builder()
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        gpu_id = (id + 1) % num_gpu
        logger.debug(f"[Worker {id}] num_gpu={num_gpu} id={id} gpu={gpu_id}")
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = "cpu"
    # input()
    net = net_builder(device)
    opp_net = net.copy()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.debug(f"[Worker {id}] Worker {id} is Ready (init_time={time.time()-st0:.3f})")
    while True:
        try:
            command, args = conn.recv()
            
            if command == 'close':
                conn.close()
                return
            
            if command == 'run':
                logger.debug(f"[Worker {id}] Start collect {int(args)} episodes")
                st0 = time.time()
                all_examples = []
                all_episode_len = []
                result_counter = ResultCounter()
                
                for e in range(int(args)):
                    env = root_env.fork()
                    state = env.reset()
                    config = copy.copy(mcts_config)
                    mcts = puct_mcts.PUCTMCTS(env, net, config)
                    train_examples = []
                    
                    while True:
                        player = env.current_player
                        # MCTS self-play
                        ########################
                        # TODO: your code here #
                        policy = None # compute policy with mcts
                        
                        symmetries = get_symmetries(state, policy) # rotate&flip the data&policy
                        train_examples += [(x[0], x[1], player) for x in symmetries]
                        
                        pass # choose a action accroding to policy
                        done = False # apply the action to env
                        if done:
                            pass # record all data
                            # tips: use env.compute_canonical_form_obs to transform the observation into BLACK's perspective
                        
                        pass # update mcts (you can use get_subtree())
                        ########################
                        
                logger.debug(f"[Worker {id}] Finished {int(args)} episodes (length={all_episode_len}) in {time.time()-st0:.3f}s, {result_counter}")
                conn.send((all_examples, result_counter))
                
            if command == 'load_net':
                file_name = str(args)
                if not file_name or file_name == 'None':
                    file_name = 'best.pth.tar'
                net.load_checkpoint(folder=checkpoint_path, filename=file_name)
                logger.debug(f"[Worker {id}] Loaded net from {checkpoint_path}")
            
            if command == 'load_opp_net':
                file_name = str(args)
                if not file_name:
                    file_name = 'temp.pth.tar'
                opp_net.load_checkpoint(folder=checkpoint_path, filename=file_name)
                logger.debug(f"[Worker {id}] Loaded net from {checkpoint_path}")
            
            if command == 'pit_opp':
                n_run = int(args)
                logger.debug(f"[Worker {id}] Start evaluating for {int(args)} round")
                last_mcts_player = PUCTPlayer(mcts_config, opp_net, deterministic=True)
                current_mcts_player = PUCTPlayer(mcts_config, net, deterministic=True)
                ret = multi_match(root_env.fork(), last_mcts_player, current_mcts_player, n_run, disable_tqdm=True)
                logger.debug(f"[Worker {id}] Finished evaluating for {int(args)} round")
                conn.send(ret)
            
            if command == 'pit_eval':
                n_run = int(args)
                logger.debug(f"[Worker {id}] Start evaluating for {int(args)} round")
                opponent = RandomPlayer()
                current_mcts_player = PUCTPlayer(mcts_config, net, deterministic=True)
                ret = multi_match(root_env.fork(), current_mcts_player, opponent, n_run, disable_tqdm=True)
                logger.debug(f"[Worker {id}] Finished evaluating for {int(args)} round")
                conn.send(ret)
                
                
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(f"[Worker {id}] shutting down worker-{id}")
            if locals().get('env_worker'):
                conn.close()
            exit(0)

class AlphaZeroParallel:
    def __init__(self, env:BaseGame, net_builder, config:AlphaZeroConfig, n_worker:int, seed:int=None):
        self.env = env
        self.net = net_builder()
        self.last_net = self.net.copy()
        self.config = config
        self.mcts_config = puct_mcts.MCTSConfig(
            C=config.C, 
            n_search=config.n_search, 
            temperature=config.temperature
        )
        assert n_worker > 0
        self.n_worker = n_worker
        self.mcts_config.with_noise = False
        self.mcts = None
        self.train_eamples_queue = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        param_list = self.env.init_param_list()
        env_builder = lambda: self.env.__class__(*param_list)
        
        if seed is None:
            seed = 0
        
        ctx = torch.multiprocessing.get_context("spawn")
        self.pipes = [ctx.Pipe() for _ in range(self.n_worker)]
        self.workers = [ctx.Process(
            target=execute_episode_worker,
            args=(
                child_conn,
                dill.dumps(env_builder, recurse=True),
                dill.dumps(net_builder, recurse=True),
                self.mcts_config,
                i,
                seed + i,
                self.config.checkpoint_path
            ),
        ) for i, (_, child_conn) in enumerate(self.pipes)]
        logger.debug(f"[AlphaZeroParallel] Created {self.n_worker} workers")
        for worker in self.workers:
            worker.start()
        logger.debug(f"[AlphaZeroParallel] Started {self.n_worker} workers")
        
    
    def execute_episode_parallel(self):
        start_search_t = time.time()
        train_examples = []
        n = self.config.n_match_train
        repeat_num = [n // self.n_worker + int(i < (n % self.n_worker)) for i in range(self.n_worker)]
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            parent_pipe.send(('load_net', ""))
            parent_pipe.send(('run', work_cnt))
        result_counter = ResultCounter()
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            result, cnt = parent_pipe.recv()
            train_examples += result
            result_counter.merge_with(cnt)
        logger.info(f"[AlphaZeroParallel] Finished {n} episodes ({len(train_examples)} examples) in {time.time()-start_search_t:.3f}s, {result_counter}")
        return train_examples
    
    def pit_with_last(self, n_run:int, opp_checkpt_filename:str, current_checkpt_filename:str):
        n_run = self.config.n_match_update
        assert n_run % 2 == 0
        n_run = n_run // 2
        repeat_num = [n_run // self.n_worker + int(i < (n_run % self.n_worker)) for i in range(self.n_worker)]
        logger.info(f"[AlphaZeroParallel] Start evaluating with last best model for {n_run*2} round")
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            parent_pipe.send(('load_net', current_checkpt_filename))
            parent_pipe.send(('load_opp_net', opp_checkpt_filename))
            parent_pipe.send(('pit_opp', work_cnt*2))
        n_p1_win, n_p2_win, n_draw = 0, 0, 0
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            r = parent_pipe.recv()[0]
            n_p1_win += r[0]
            n_p2_win += r[1]
            n_draw += r[2]
        return (n_p1_win, n_p2_win, n_draw)
    
    def evaluate(self):
        n_run = self.config.n_match_eval
        assert n_run % 2 == 0
        n_run = n_run // 2
        repeat_num = [n_run // self.n_worker + int(i < (n_run % self.n_worker)) for i in range(self.n_worker)]
        logger.info(f"[AlphaZeroParallel] Start evaluating with baseline for {n_run*2} round")
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            parent_pipe.send(('load_net', ""))
            parent_pipe.send(('pit_eval', work_cnt*2))
        class ans_pack:
            def __init__(self, r=(0, 0, 0)):
                self.n_p1_win = r[0]
                self.n_p2_win = r[1]
                self.n_draw = r[2]
            def add(self, r):
                self.n_p1_win += r[0]
                self.n_p2_win += r[1]
                self.n_draw += r[2]
            def tolist(self):
                return self.n_p1_win, self.n_p2_win, self.n_draw
        total_result = ans_pack()
        first_result = ans_pack()
        latter_result = ans_pack()
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            r, first, latter = parent_pipe.recv()
            total_result.add(r)
            first_result.add(first)
            latter_result.add(latter)
        result = [total_result.tolist(), first_result.tolist(), latter_result.tolist()]
        logger.info(f"[EVALUATION RESULT]: win{result[0][0]}, lose{result[0][1]}, draw{result[0][2]}")
        logger.info(f"[EVALUATION RESULT]:(first)  win{result[1][0]}, lose{result[1][1]}, draw{result[1][2]}")
        logger.info(f"[EVALUATION RESULT]:(second) win{result[2][0]}, lose{result[2][1]}, draw{result[2][2]}")
    
    def learn(self):
        self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='best.pth.tar')
        for iter in range(1, self.config.n_train_iter + 1):
            st = time.time()
            logger.info(f"------ Start Self-Play Iteration {iter} ------")
            
            # collect new examples
            self.train_eamples_queue += self.execute_episode_parallel()
            
            # pop old examples
            if len(self.train_eamples_queue) > self.config.max_queue_length:
                self.train_eamples_queue = self.train_eamples_queue[-self.config.max_queue_length:]
            
            # shuffle examples for training
            train_data = copy.copy(self.train_eamples_queue)
            shuffle(train_data)
            logger.info(f"[TRAIN DATA SIZE]: {len(train_data)}")
            
            # with open(os.path.join(self.config.checkpoint_path, f'buffer_iter-{iter:04d}.pickle'), 'wb') as handle:
            #     pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # train current net
            self.net.train(train_data)
            
            # evaluate current net
            self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')
            
            result = self.pit_with_last(self.config.n_match_update, 'best.pth.tar', 'temp.pth.tar')
            # win_rate = result[1] / sum(result)
            total_win_lose = result[0] + result[1]
            win_rate = result[1] / total_win_lose if total_win_lose > 0 else 1
            logger.info(f"[EVALUATION RESULT]: currrent_win{result[1]}, last_win{result[0]}, draw{result[2]}; win_rate={win_rate:.3f}")
            
            if win_rate > self.config.update_threshold:
                self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='best.pth.tar')
                logger.info(f"[ACCEPT NEW MODEL]")
                self.evaluate()
            else:
                self.net.load_checkpoint(folder=self.config.checkpoint_path, filename='best.pth.tar')
                logger.info(f"[REJECT NEW MODEL]")
            logger.info(f"------ Finished Self-Play Iteration {iter} in {time.time()-st:.3f}s ------\n")

    def close(self):
        for worker in self.workers:
            worker.close()

if __name__ == "__main__":
    from env import *
    import torch
    
    MASTER_SEED = 1
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    torch.manual_seed(MASTER_SEED)
    
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("log.txt") # Use this to record your logs in file 
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    
    # Linear
    config = AlphaZeroConfig(
        n_train_iter=50,
        n_match_train=20,
        n_match_update=20,
        n_match_eval=20,
        max_queue_length=160000,
        update_threshold=0.000,
        n_search=240, 
        temperature=1.0, 
        C=1.0,
        checkpoint_path="checkpoint/linear_7x7_exfeat_norm"
    )
    model_training_config = NumpyModelTrainingConfig(
        epochs=10,
        batch_size=128,
        lr=0.0001,
        weight_decay=0.001
    )
    assert config.n_match_update % 2 == 0
    assert config.n_match_eval % 2 == 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = GoGame(7, obs_mode="extra_feature")
    # env = GoGame(7) # oringin observation, only the n*n game board
    
    def base_function(X: np.ndarray) -> np.ndarray:
        return X
    
    def net_builder(device=device):
        net = NumpyLinearModel(env.observation_size, env.action_space_size, None, device=device, base_function=base_function)
        net = NumpyLinearModelTrainer(env.observation_size, env.action_space_size, net, model_training_config)
        return net
        
    N_WORKER = 10 # Set a smaller worker number when debug
    alphazero = AlphaZeroParallel(env, net_builder, config, N_WORKER, seed=MASTER_SEED)
    alphazero.learn()