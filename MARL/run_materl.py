from MATERL_PPO_MA import MATERL_PPO_MA
from common.utils import agg_double_list, copy_file_ppo, init_dir
from datetime import datetime

import argparse
import configparser
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import gymnasium as gym
except Exception:
    import gym

import sys
sys.path.append("../highway-env")
import highway_env  # noqa: F401


def parse_args():
    """
    Description for this experiment:
        + TERL on merge-multi-agent-v0
        + default: globalR, seed = 0
    """
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs_materl.ini'
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment using MATERL'))
    parser.add_argument('--base-dir', type=str, required=False, default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False, default='train', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False, default=default_config_dir, help="experiment config path")
    parser.add_argument('--model-dir', type=str, required=False, default='', help="pretrained model path")
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    args = parser.parse_args()
    return args


def train(args):
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # create an experiment folder (align baseline)
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = base_dir + now
    dirs = init_dir(output_dir)
    copy_file_ppo(dirs['configs'])

    # choose model_dir (align baseline)
    if os.path.exists(args.model_dir):
        model_dir = args.model_dir
    else:
        model_dir = dirs['models']

    # ======== ENV (align baseline; explicit per-key injection) ========
    env = gym.make('merge-multi-agent-v0')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    assert env.T % ROLL_OUT_N_STEPS == 0

    env_eval = gym.make('merge-multi-agent-v0')
    env_eval.config['seed'] = config.getint('ENV_CONFIG', 'seed') + 1
    env_eval.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env_eval.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env_eval.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env_eval.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env_eval.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env_eval.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env_eval.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env_eval.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env_eval.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env_eval.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    # ======== dimensions (align baseline) + lightweight self-check (do not assert) ========
    state_dim = env.n_s
    action_dim = env.n_a
    _obs0 = env.reset()
    if isinstance(_obs0, tuple): _obs0 = _obs0[0]
    # self-check with first agent's vector length (warn only)
    if isinstance(_obs0, np.ndarray):
        _len = int(np.asarray(_obs0[0] if _obs0.ndim == 2 else _obs0, dtype=np.float32).reshape(-1).size)
    elif isinstance(_obs0, (list, tuple)):
        _len = int(np.asarray(_obs0[0], dtype=np.float32).reshape(-1).size)
    else:
        _len = int(np.asarray(_obs0, dtype=np.float32).reshape(-1).size)
    if _len != state_dim:
        print(f"[warn] env.n_s={state_dim} but first-agent obs={_len}. Proceeding with env.n_s to stay comparable with baselines.")

    # seeds for evaluation (align baseline via CLI or config)
    cfg_seeds = config.get('TRAIN_CONFIG', 'test_seeds', fallback=args.evaluation_seeds)
    test_seeds = cfg_seeds if cfg_seeds else args.evaluation_seeds

    # ======== Hyper-params (names aligned) ========
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'max_grad_norm')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'entropy_reg')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    actor_lr = config.getfloat('MODEL_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('MODEL_CONFIG', 'critic_lr')
    gae_tau = config.getfloat('MODEL_CONFIG', 'gae_tau')
    clip_param = config.getfloat('MODEL_CONFIG', 'clip_param')
    k_epochs = config.getint('MODEL_CONFIG', 'k_epochs')

    # train configs
    MAX_EPISODES = config.getint('TRAIN_CONFIG', 'MAX_EPISODES')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    EVAL_INTERVAL = config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL')
    EVAL_EPISODES = config.getint('TRAIN_CONFIG', 'EVAL_EPISODES')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')
    save_interval = config.getint('TRAIN_CONFIG', 'save_interval', fallback=500)

    # TERL configs
    k_max = config.getint('TERL_CONFIG','k_max')
    rollout_sensitivity = config.getfloat('TERL_CONFIG','rollout_sensitivity')
    loss_state_th = config.getfloat('TERL_CONFIG','loss_state_th')
    reward_loss_th = config.getfloat('TERL_CONFIG','reward_loss_th')
    use_knowledge = config.getboolean('TERL_CONFIG','use_knowledge')
    knowledge_bias_scale = config.getfloat('TERL_CONFIG','knowledge_bias_scale')

    # ======== Build Agent ========
    materl = MATERL_PPO_MA(env=env,
                           state_dim=state_dim, action_dim=action_dim,
                           gamma=reward_gamma, clip=clip_param,
                           actor_lr=actor_lr, critic_lr=critic_lr,
                           lam=gae_tau, k_epochs=k_epochs, ent_coef=ENTROPY_REG,
                           max_grad_norm=MAX_GRAD_NORM,
                           k_max=k_max, rollout_sensitivity=rollout_sensitivity,
                           loss_state_th=loss_state_th, reward_loss_th=reward_loss_th,
                           use_knowledge=use_knowledge, knowledge_bias_scale=knowledge_bias_scale,
                           action_masking=config.getboolean('MODEL_CONFIG','action_masking'),
                           # 新增：传入评估 seeds 与 traffic_density，供 evaluation() 对齐基线 reset
                           test_seeds=test_seeds,
                           traffic_density=config.getint('ENV_CONFIG', 'traffic_density'))

    # load the model if exist (directory style)
    materl.load(model_dir, train_mode=True)

    eval_rewards = []
    while materl.n_episodes < MAX_EPISODES:
        # ===== 跨边界补评：记录调用前的 episode 数 =====
        prev_epi = materl.n_episodes

        # collect + (inside) fit fake env + virtual rollout (TERL)
        materl.interact(rollout_steps=ROLL_OUT_N_STEPS,
                        reward_scale=reward_scale, reward_type=reward_type)

        if materl.n_episodes >= EPISODES_BEFORE_TRAIN:
            materl.train()

        # ===== NEW: 补评逻辑（防止一次 rollout 跨过多个评估点）=====
        curr_epi = materl.n_episodes
        for e in range(prev_epi + 1, curr_epi + 1):
            if e % EVAL_INTERVAL == 0:
                rewards, _, _, _ = materl.evaluation(env_eval, dirs['train_videos'], EVAL_EPISODES)
                rewards_mu = float(np.mean([r[0] for r in rewards])) if rewards else 0.0
                print(f"Episode {e}, Average Reward {rewards_mu:.2f}")
                eval_rewards.append(rewards_mu)
                materl.save(dirs['models'], e)

        # 定期额外保存（与基线一致）
        if (materl.n_episodes + 1) % save_interval == 0:
            materl.save(dirs['models'], materl.n_episodes + 1)

    materl.save(dirs['models'], MAX_EPISODES + 2)


def evaluate(args):
    if os.path.exists(args.model_dir):
        model_dir = args.model_dir + '/models/'
    else:
        raise Exception("Sorry, no pretrained models")
    config_dir = args.model_dir + '/configs/configs_materl.ini'
    config = configparser.ConfigParser()
    config.read(config_dir)

    video_dir = args.model_dir + '/eval_videos'

    # model configs
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'max_grad_norm')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'entropy_reg')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    actor_lr = config.getfloat('MODEL_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('MODEL_CONFIG', 'critic_lr')
    gae_tau = config.getfloat('MODEL_CONFIG', 'gae_tau')
    clip_param = config.getfloat('MODEL_CONFIG', 'clip_param')
    k_epochs = config.getint('MODEL_CONFIG', 'k_epochs')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')

    # train configs
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')
    test_seeds = config.get('TRAIN_CONFIG', 'test_seeds', fallback=','.join([str(i) for i in range(0, 600, 20)]))

    # env
    env = gym.make('merge-multi-agent-v0')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    assert env.T % ROLL_OUT_N_STEPS == 0
    state_dim = env.n_s
    action_dim = env.n_a

    materl = MATERL_PPO_MA(env=env,
                           state_dim=state_dim, action_dim=action_dim,
                           gamma=reward_gamma, clip=clip_param,
                           actor_lr=actor_lr, critic_lr=critic_lr,
                           lam=gae_tau, k_epochs=k_epochs, ent_coef=ENTROPY_REG,
                           max_grad_norm=MAX_GRAD_NORM,
                           k_max=config.getint('TERL_CONFIG','k_max'),
                           rollout_sensitivity=config.getfloat('TERL_CONFIG','rollout_sensitivity'),
                           loss_state_th=config.getfloat('TERL_CONFIG','loss_state_th'),
                           reward_loss_th=config.getfloat('TERL_CONFIG','reward_loss_th'),
                           use_knowledge=config.getboolean('TERL_CONFIG','use_knowledge'),
                           knowledge_bias_scale=config.getfloat('TERL_CONFIG','knowledge_bias_scale'),
                           action_masking=config.getboolean('MODEL_CONFIG','action_masking'),
                           test_seeds=test_seeds,
                           traffic_density=config.getint('ENV_CONFIG', 'traffic_density'))

    materl.load(model_dir, train_mode=False)
    rewards, _, _, _ = materl.evaluation(env, video_dir, 3)


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
