import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from common.utils import VideoRecorder  # 基线相同的录像工具

class DualReplayBuffer:
    def __init__(self, capacity=200000):
        self.real, self.virtual = [], []
        self.capacity = capacity
    def push_real(self, s,a,r,ns,done):
        if len(self.real) >= self.capacity: self.real.pop(0)
        self.real.append((s,a,r,ns,done))
    def push_virtual(self, s,a,r,ns,done):
        if len(self.virtual) >= self.capacity: self.virtual.pop(0)
        self.virtual.append((s,a,r,ns,done))
    def sample_mixed(self, n_real, n_virtual):
        import random
        r = random.sample(self.real, min(n_real, len(self.real))) if self.real else []
        v = random.sample(self.virtual, min(n_virtual, len(self.virtual))) if self.virtual else []
        batch = r + v
        random.shuffle(batch)
        if not batch: return None
        s = np.stack([b[0] for b in batch]); a = np.stack([b[1] for b in batch])
        r = np.stack([b[2] for b in batch]).astype(np.float32)
        ns = np.stack([b[3] for b in batch]); d = np.stack([b[4] for b in batch]).astype(np.float32)
        return s,a,r,ns,d
    def last_real_batch(self, n): return self.real[-n:] if self.real else []

class FakeEnvModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(128,128), lr=1e-3, device="cpu"):
        super().__init__()
        self.state_dim = state_dim; self.action_dim = action_dim
        self.device = torch.device(device)
        def mlp(in_dim, out_dim):
            layers = []; last = in_dim
            for h in hidden: layers += [nn.Linear(last, h), nn.Tanh()]; last = h
            layers += [nn.Linear(last, out_dim)]; return nn.Sequential(*layers)
        self.state_net = mlp(state_dim + action_dim, state_dim).to(self.device)
        self.rew_net   = mlp(state_dim + action_dim, 1).to(self.device)
        self.opt = torch.optim.Adam(list(self.state_net.parameters()) + list(self.rew_net.parameters()), lr=lr)
    @torch.no_grad()
    def step(self, state, action):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        x = torch.cat([s, a], dim=-1)
        ds = self.state_net(x); r  = self.rew_net(x)
        ns = (s + ds).squeeze(0).cpu().numpy()
        r  = r.squeeze(0).squeeze(-1).item()
        return ns, r
    def train_supervised(self, batch, iters=2, bs=64):
        if len(batch) == 0: return 1e9, 1e9
        import random
        losses_s, losses_r = [], []
        for _ in range(iters):
            random.shuffle(batch)
            for i in range(0, len(batch), bs):
                chunk = batch[i:i+bs]
                s = torch.as_tensor(np.stack([c[0] for c in chunk]), dtype=torch.float32, device=self.device)
                a = torch.as_tensor(np.stack([c[1] for c in chunk]), dtype=torch.float32, device=self.device)
                r = torch.as_tensor(np.stack([c[2] for c in chunk]), dtype=torch.float32, device=self.device).unsqueeze(-1)
                ns= torch.as_tensor(np.stack([c[3] for c in chunk]), dtype=torch.float32, device=self.device)
                x = torch.cat([s,a], dim=-1)
                ds_pred = self.state_net(x); r_pred  = self.rew_net(x)
                ds_tgt  = ns - s
                loss_s  = ((ds_pred - ds_tgt)**2).mean(); loss_r  = ((r_pred - r)**2).mean()
                loss = loss_s + loss_r
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                losses_s.append(loss_s.item()); losses_r.append(loss_r.item())
        return float(np.mean(losses_s)), float(np.mean(losses_r))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, action_dim))
    def forward(self, s): return self.net(s)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))
    def forward(self, s): return self.net(s)

class MATERL_PPO_MA:
    def __init__(self, env, state_dim, action_dim,
                 gamma=0.99, clip=0.2, actor_lr=3e-4, critic_lr=3e-4, lam=0.95,
                 k_epochs=4, ent_coef=0.01, max_grad_norm=5.0,
                 k_max=100, rollout_sensitivity=2.0,
                 loss_state_th=1e-2, reward_loss_th=1.0,
                 use_knowledge=False, knowledge_bias_scale=0.2,
                 action_masking=False, device="cpu",
                 # 新增：对齐基线评测的参数
                 test_seeds="0,25,50,75", traffic_density=2):
        self.env=env
        self.state_dim=state_dim; self.action_dim=action_dim
        self.device=torch.device(device)
        self.actor=Actor(state_dim, action_dim).to(self.device)
        self.critic=Critic(state_dim).to(self.device)
        self.opt_actor=torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_critic=torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma=gamma; self.clip=clip; self.lam=lam; self.k_epochs=k_epochs
        self.ent_coef=ent_coef; self.max_grad_norm=max_grad_norm
        self.buffer=DualReplayBuffer()
        self.fake=FakeEnvModel(state_dim, action_dim, device=self.device)
        self.k_max=k_max; self.rollout_sensitivity=rollout_sensitivity
        self.loss_state_th=loss_state_th; self.reward_loss_th=reward_loss_th
        self.use_knowledge=use_knowledge; self.knowledge_bias_scale=knowledge_bias_scale
        self.knowledge = None
        self.action_masking = action_masking
        self.n_episodes = 0
        self.episode_done = False
        self._last_info = {}
        # 评测辅助
        self.test_seeds = test_seeds
        self.traffic_density = traffic_density

    # --- 观测+掩码解析 ---
    def _obs_to_list(self, obs):
        if isinstance(obs, tuple):  # (obs, action_mask) 在此不取 mask
            obs = obs[0]
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                return [obs.astype(np.float32).reshape(-1)]
            elif obs.ndim == 2:
                return [obs[i].astype(np.float32).reshape(-1) for i in range(obs.shape[0])]
            else:
                raise ValueError(f"Unsupported obs ndarray shape: {obs.shape}")
        if isinstance(obs, (list, tuple)):
            return [np.asarray(o, dtype=np.float32).reshape(-1) for o in obs]
        return [np.asarray(obs, dtype=np.float32).reshape(-1)]

    def _split_obs_mask(self, out_obs):
        """返回 (obs_list, mask_list 或 None)。兼容 reset/step 可能返回 (obs, mask) 的情况。"""
        mask_list = None
        obs = out_obs
        if isinstance(out_obs, tuple) and len(out_obs) == 2:
            obs, mask = out_obs
            try:
                mask_arr = np.asarray(mask)
                if mask_arr.ndim == 1:
                    mask_list = [mask_arr.astype(bool)]
                elif mask_arr.ndim == 2:
                    mask_list = [mask_arr[i].astype(bool) for i in range(mask_arr.shape[0])]
            except Exception:
                mask_list = None
        return self._obs_to_list(obs), mask_list

    def _select_actions(self, obs_list, mask_list=None):
        s = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        logits = self.actor(s)
        # 可选：无效动作屏蔽
        if self.action_masking and (mask_list is not None):
            try:
                m = torch.as_tensor(np.stack(mask_list), dtype=torch.bool, device=self.device)
                logits = logits.masked_fill(~m, -1e9)
            except Exception:
                pass
        dist = Categorical(logits=logits)
        a = dist.sample(); logp = dist.log_prob(a)
        return a.detach().cpu().numpy().tolist(), logp.detach().cpu().numpy().tolist()

    def interact(self, rollout_steps=100, reward_scale=20.0, reward_type="global_R"):
        """收集真实数据 -> 拟合 fake env -> 生成虚拟数据。注意：存入记忆前对奖励做基线式缩放：/ reward_scale。"""
        steps = 0
        real_samples = virt_samples = 0
        ep_returns = []

        out = self.env.reset()
        obs_list, mask_list = self._split_obs_mask(out)
        ep_ret = 0.0; done = False
        any_episode_done = False

        while steps < rollout_steps:
            actions, _ = self._select_actions(obs_list, mask_list)
            out = self.env.step(tuple(actions))
            if isinstance(out, tuple) and len(out) == 5:
                next_out_obs, rew, terminated, truncated, info = out; done = bool(terminated or truncated)
            else:
                next_out_obs, rew, done, info = out

            next_obs_list, next_mask_list = self._split_obs_mask(next_out_obs)

            # 回报聚合（global_R 用标量/均值）
            if isinstance(rew, (list, tuple, np.ndarray)):
                rew_list = list(np.asarray(rew, dtype=np.float32).reshape(-1))
                r_scalar = float(np.mean(rew_list))
            else:
                r_scalar = float(rew)

            if reward_type.lower() == "global_r":
                per_agent_reward = [r_scalar] * len(next_obs_list)
            else:
                # 若环境返回逐 agent 列表，才使用；否则沿用 r_scalar
                per_agent_reward = [r_scalar] * len(next_obs_list)

            # === 存入记忆前：对齐基线，奖励 / reward_scale ===
            for i in range(len(obs_list)):
                a_onehot = np.zeros(self.action_dim, dtype=np.float32)
                a_onehot[actions[i]] = 1.0
                r_scaled = per_agent_reward[i] / float(reward_scale)
                self.buffer.push_real(obs_list[i], a_onehot, r_scaled, next_obs_list[i], float(not done))
                real_samples += 1

            ep_ret += r_scalar
            obs_list, mask_list = next_obs_list, next_mask_list
            steps += 1

            if done:
                self.n_episodes += 1
                ep_returns.append(ep_ret); ep_ret = 0.0
                out = self.env.reset()
                obs_list, mask_list = self._split_obs_mask(out)
                done = False
                any_episode_done = True

        # 拟合 fake env
        batch = self.buffer.last_real_batch(2048)
        fake_s_loss, fake_r_loss = self.fake.train_supervised(batch, iters=2, bs=128)

        # 虚拟 rollout（阈值满足才开）
        k_star = 0
        if fake_s_loss < self.loss_state_th and fake_r_loss < self.reward_loss_th:
            k_star = min(self.k_max, int(self.rollout_sensitivity / max(fake_s_loss, 1e-6)))
            seed_batch = self.buffer.last_real_batch(max(1, k_star))
            for s,a,r,ns,d in seed_batch:
                cur_s = s
                for _ in range(k_star):
                    st = torch.as_tensor(cur_s, dtype=torch.float32, device=self.device).unsqueeze(0)
                    logits = self.actor(st)
                    aa = int(torch.argmax(logits, dim=-1).item())
                    onehot = np.zeros(self.action_dim, dtype=np.float32); onehot[aa] = 1.0
                    nss, rr = self.fake.step(cur_s, onehot)
                    # 虚拟样本也按同尺度缩放（/ reward_scale）
                    self.buffer.push_virtual(cur_s, onehot, rr / float(reward_scale), nss, 1.0)
                    cur_s = nss; virt_samples += 1

        self._last_info = dict(
            ep_ret_mean = float(np.mean(ep_returns)) if ep_returns else 0.0,
            fake_state_loss = fake_s_loss,
            fake_reward_loss = fake_r_loss,
            k_star = int(k_star),
            real_samples = int(real_samples),
            virt_samples = int(virt_samples)
        )
        self.episode_done = any_episode_done

    def train(self, real_batch_n=4096, virtual_batch_n=4096):
        B = self.buffer.sample_mixed(real_batch_n, virtual_batch_n) or self.buffer.sample_mixed(real_batch_n, 0)
        if B is None: return
        s,a,r,ns,d = B
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.argmax(a, axis=1), dtype=torch.long, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)  # 已经是 /scale 后的
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)
        with torch.no_grad():
            v = self.critic(s); next_v = self.critic(torch.as_tensor(ns, dtype=torch.float32, device=self.device))
            td_target = r + self.gamma * next_v * d
            adv = (td_target - v); adv = (adv - adv.mean())/(adv.std()+1e-8)
        old_logits = self.actor(s).detach(); old_dist = Categorical(logits=old_logits)
        old_logp = old_dist.log_prob(a).detach()
        for _ in range(self.k_epochs):
            logits = self.actor(s); dist = Categorical(logits=logits)
            logp = dist.log_prob(a).unsqueeze(-1)
            ratio = torch.exp(logp - old_logp.unsqueeze(-1))
            surr1 = ratio * adv; surr2 = torch.clamp(ratio, 1.0-self.clip, 1.0+self.clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef*dist.entropy().mean()
            self.opt_actor.zero_grad(); actor_loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.opt_actor.step()
            v_pred = self.critic(s); critic_loss = ((v_pred-td_target.detach())**2).mean()
            self.opt_critic.zero_grad(); critic_loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.opt_critic.step()

    # === 基线式录像 + reset 对齐（testing_seeds/num_CAV）===
    def _num_cav_from_density(self, density):
        # 参考基线的做法：这里给一个保守近似映射（如有专门函数，可替换）
        try:
            d = int(density)
        except Exception:
            d = 2
        table = {1: 2, 2: 3, 3: 4}
        return table.get(d, 3)

    def evaluation(self, env_eval, video_dir, eval_episodes, is_train=True):
        import os
        os.makedirs(video_dir, exist_ok=True)

        # 离屏渲染，便于录帧
        if hasattr(env_eval, "config"):
            env_eval.config['offscreen_rendering'] = True

        # seeds（从构造函数传入的 test_seeds 字符串）
        try:
            seeds = [int(s) for s in str(self.test_seeds).split(',') if str(s).strip() != ""]
        except Exception:
            seeds = [0]

        num_cav = self._num_cav_from_density(self.traffic_density)
        rewards = []
        video_recorder = None

        for i in range(eval_episodes):
            # 对齐基线：携带 testing_seeds / num_CAV 重置（失败安全回退）
            try:
                out = env_eval.reset(is_training=False, testing_seeds=seeds[i % len(seeds)], num_CAV=num_cav)
            except Exception:
                out = env_eval.reset()

            obs_list, mask_list = self._split_obs_mask(out)
            done = False
            ep_ret = 0.0

            # 首帧 + 初始化录像器（与基线一致）
            try:
                rendered_frame = env_eval.render(mode="rgb_array")
                video_filename = os.path.join(video_dir, f"testing_episode{self.n_episodes + 1}_{i}.mp4")
                video_recorder = VideoRecorder(video_filename,
                                               frame_size=rendered_frame.shape, fps=5)
                video_recorder.add_frame(rendered_frame)
            except Exception:
                video_recorder = None

            while not done:
                actions, _ = self._select_actions(obs_list, mask_list)
                out = env_eval.step(tuple(actions))
                if isinstance(out, tuple) and len(out) == 5:
                    next_out_obs, rew, terminated, truncated, info = out
                    done = bool(terminated or truncated)
                else:
                    next_out_obs, rew, done, info = out

                next_obs_list, next_mask_list = self._split_obs_mask(next_out_obs)

                # 录一帧
                if video_recorder is not None:
                    try:
                        rendered_frame = env_eval.render(mode="rgb_array")
                        video_recorder.add_frame(rendered_frame)
                    except Exception:
                        pass

                # 回报：用标量（或多体均值）与训练保持同口径
                if isinstance(rew, (list, tuple, np.ndarray)):
                    rew_list = list(np.asarray(rew, dtype=np.float32).reshape(-1))
                    r_scalar = float(np.mean(rew_list))
                else:
                    r_scalar = float(rew)
                ep_ret += r_scalar

                obs_list, mask_list = next_obs_list, next_mask_list

            if video_recorder is not None:
                video_recorder.release()
                video_recorder = None

            rewards.append([ep_ret])

        try:
            env_eval.close()
        except Exception:
            pass
        return rewards, [], [], []

    def save(self, models_dir, step):
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, f"materl_ep{int(step)}.pt")
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "fake_state": self.fake.state_dict(),
            "n_episodes": self.n_episodes
        }, path)

    def load(self, model_dir, train_mode=True):
        if os.path.isdir(model_dir):
            pts = sorted(glob.glob(os.path.join(model_dir, "*.pt")))
            if not pts:
                return
            path = pts[-1]
        else:
            path = model_dir
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
        self.fake.load_state_dict(ckpt["fake_state"])
        self.n_episodes = ckpt.get("n_episodes", 0)

    def last_info(self):
        return dict(self._last_info)
