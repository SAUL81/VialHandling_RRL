import datetime
import gymnasium as gym
import os
import os.path as osp
import pickle
import numpy as np
import itertools
import torch
import moviepy.editor as mpy
import cv2
from torch import nn, optim

from sac import SAC
from replay_memory import ReplayMemory, ConstraintReplayMemory
from environment import VialHandlingEnv
from utils import linear_schedule #, recovery_config_setup


TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def torchify(x): return torch.FloatTensor(x).to('cuda')


def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


# Process observation for CNN
def process_obs(obs, env_name):
    if 'extraction' in env_name:
        obs = cv2.resize(obs, (64, 48), interpolation=cv2.INTER_AREA)
    im = np.transpose(obs, (2, 0, 1))
    return im


class Experiment:
    def __init__(self, exp_cfg):
        self.exp_cfg = exp_cfg
        # Logging setup
        self.logdir = os.path.join(
            self.exp_cfg.logdir, '{}_SAC_{}_{}_{}'.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                self.exp_cfg.env_name, self.exp_cfg.policy,
                self.exp_cfg.logdir_suffix))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        print("LOGDIR: ", self.logdir)
        pickle.dump(self.exp_cfg,
                    open(os.path.join(self.logdir, "args.pkl"), "wb"))

        # Experiment setup
        self.experiment_setup()

        # Memory
        self.memory = ReplayMemory(self.exp_cfg.replay_size, self.exp_cfg.seed)
        self.recovery_memory = ConstraintReplayMemory(
            self.exp_cfg.safe_replay_size, self.exp_cfg.seed)
        self.all_ep_data = []

        self.total_numsteps = 0
        self.updates = 0
        self.num_constraint_violations = 0
        self.num_unsafe_transitions = 0

        self.num_viols = 0
        self.num_successes = 0
        self.viol_and_recovery = 0
        self.viol_and_no_recovery = 0
        # Get demos
        self.task_demos = self.exp_cfg.task_demos
        self.constraint_demo_data, self.task_demo_data, self.obs_seqs, self.ac_seqs, self.constraint_seqs = self.get_offline_data(
        )

        # Get multiplier schedule for RSPO
        if self.exp_cfg.nu_schedule:
            self.nu_schedule = linear_schedule(self.exp_cfg.nu_start,
                                               self.exp_cfg.nu_end,
                                               self.exp_cfg.num_eps)
        else:
            self.nu_schedule = linear_schedule(self.exp_cfg.nu,
                                               self.exp_cfg.nu, 0)

    def experiment_setup(self):
        torch.manual_seed(self.exp_cfg.seed)
        np.random.seed(self.exp_cfg.seed)
        recovery_policy = None
        env = VialHandlingEnv()
        self.env = env 
        self.recovery_policy = recovery_policy
        self.env.seed(self.exp_cfg.seed)
        self.env.action_space.seed(self.exp_cfg.seed)
        agent = self.agent_setup(env)
        self.agent = agent

    def agent_setup(self, env):
        agent = SAC(env.observation_space,
                    env.action_space,
                    self.exp_cfg,
                    self.logdir,
                    tmp_env=VialHandlingEnv())
        return agent

    def get_offline_data(self):
        # Get demonstrations
        task_demo_data = None
        obs_seqs = []
        ac_seqs = []
        constraint_seqs = []
        if not self.exp_cfg.task_demos:
            if self.exp_cfg.env_name == 'reacher':
                constraint_demo_data = pickle.load(
                    open(
                        osp.join("demos", "dvrk_reach",
                                 "constraint_demos.pkl"), "rb"))
                if self.exp_cfg.cnn:
                    constraint_demo_data = constraint_demo_data['images']
                else:
                    constraint_demo_data = constraint_demo_data['lowdim']
            elif 'maze' in self.exp_cfg.env_name:
                # Maze
                if self.exp_cfg.env_name == 'maze':
                    constraint_demo_data = pickle.load(
                        open(
                            osp.join("demos", self.exp_cfg.env_name,
                                     "constraint_demos.pkl"), "rb"))
                else:
                    # Image Maze
                    demo_data = pickle.load(
                        open(
                            osp.join("demos", self.exp_cfg.env_name,
                                     "demos.pkl"), "rb"))
                    constraint_demo_data = demo_data['constraint_demo_data']
                    obs_seqs = demo_data['obs_seqs']
                    ac_seqs = demo_data['ac_seqs']
                    constraint_seqs = demo_data['constraint_seqs']
            elif 'extraction' in self.exp_cfg.env_name:
                # Object Extraction, Object Extraction (Dynamic Obstacle)
                folder_name = self.exp_cfg.env_name.split('_env')[0]
                constraint_demo_data = pickle.load(
                    open(
                        osp.join("demos", folder_name, "constraint_demos.pkl"),
                        "rb"))
                
                """for our environment"""
            elif 'vial_handling' in self.exp_cfg.env_name:
                constraint_demo_data = pickle.load(
                    open("constraint_demos.pkl", "rb"))
                """end"""

            else:
                # Navigation 1 and 2
                constraint_demo_data = self.env.transition_function(
                    self.exp_cfg.num_unsafe_transitions)
        else:
            if 'extraction' in self.exp_cfg.env_name:
                folder_name = self.exp_cfg.env_name.split('_env')[0]
                task_demo_data = pickle.load(
                    open(osp.join("demos", folder_name, "task_demos.pkl"),
                         "rb"))
                constraint_demo_data = pickle.load(
                    open(
                        osp.join("demos", folder_name, "constraint_demos.pkl"),
                        "rb"))
                # Get all violations in front to get as many violations as
                # possible
                constraint_demo_data_list_safe = []
                constraint_demo_data_list_viol = []
                for i in range(len(constraint_demo_data)):
                    if constraint_demo_data[i][2] == 1:
                        constraint_demo_data_list_viol.append(
                            constraint_demo_data[i])
                    else:
                        constraint_demo_data_list_safe.append(
                            constraint_demo_data[i])

                constraint_demo_data = constraint_demo_data_list_viol[:int(
                    0.5 * self.exp_cfg.num_unsafe_transitions
                )] + constraint_demo_data_list_safe

                """for our environment"""
            elif 'vial_handling' in self.exp_cfg.env_name:
                task_demo_data = pickle.load(
                    open("task_demos.pkl", "rb"))
                constraint_demo_data = pickle.load(
                    open("constraint_demos.pkl", "rb"))
                
                # Get all violations in front to get as many violations as
                # possible
                constraint_demo_data_list_safe = []
                constraint_demo_data_list_viol = []
                for i in range(len(constraint_demo_data)):
                    if constraint_demo_data[i][2] == 1.0:
                        constraint_demo_data_list_viol.append(
                            constraint_demo_data[i])
                    else:
                        constraint_demo_data_list_safe.append(
                            constraint_demo_data[i])

                constraint_demo_data = constraint_demo_data_list_viol[:int(
                    0.5 * self.exp_cfg.num_unsafe_transitions
                )] + constraint_demo_data_list_safe
                """end"""

            else:
                constraint_demo_data, task_demo_data = self.env.transition_function(
                    self.exp_cfg.num_unsafe_transitions, task_demos=self.exp_cfg.task_demos)
        return constraint_demo_data, task_demo_data, obs_seqs, ac_seqs, constraint_seqs

    def train_MB_recovery(self, states, actions, next_states=None, epochs=50):
        if next_states is not None:
            self.recovery_policy.train(states,
                                       actions,
                                       random=True,
                                       next_obs=next_states,
                                       epochs=epochs)
        else:
            self.recovery_policy.train(states, actions)

    def pretrain_critic_recovery(self):
        if not self.exp_cfg.vismpc_recovery:
            # Get data for recovery policy and safety critic training
            demo_data_states = np.array([
                d[0] for d in
                self.constraint_demo_data[:self.exp_cfg.num_unsafe_transitions]
            ])
            demo_data_actions = np.array([
                d[1] for d in
                self.constraint_demo_data[:self.exp_cfg.num_unsafe_transitions]
            ])
            demo_data_next_states = np.array([
                d[3] for d in
                self.constraint_demo_data[:self.exp_cfg.num_unsafe_transitions]
            ])
            self.num_unsafe_transitions = 0
            for transition in self.constraint_demo_data:
                self.recovery_memory.push(*transition)
                self.num_constraint_violations += int(transition[2])
                self.num_unsafe_transitions += 1
                if self.num_unsafe_transitions == self.exp_cfg.num_unsafe_transitions:
                    break
            print("Number of Constraint Transitions: ",
                  self.num_unsafe_transitions)
            print("Number of Constraint Violations: ",
                  self.num_constraint_violations)

            # Train DDPG recovery policy
            for i in range(self.exp_cfg.critic_safe_pretraining_steps):
                if i % 100 == 0:
                    print("CRITIC SAFE UPDATE STEP: ", i)
                self.agent.safety_critic.update_parameters(
                    memory=self.recovery_memory,
                    policy=self.agent.policy,
                    batch_size=min(self.exp_cfg.batch_size,
                                   len(self.constraint_demo_data)))

            # Train PETS recovery policy
            if not (self.exp_cfg.MF_recovery
                    or self.exp_cfg.Q_sampling_recovery
                    or self.exp_cfg.DGD_constraints or self.exp_cfg.RCPO):
                self.train_MB_recovery(demo_data_states,
                                  demo_data_actions,
                                  demo_data_next_states,
                                  epochs=50)
        else:
            # Pre-train vis dynamics model if needed
            if not self.exp_cfg.load_vismpc:
                self.recovery_policy.train(self.obs_seqs,
                                           self.ac_seqs,
                                           self.constraint_seqs,
                                           num_train_steps=20000)
            # Get data for recovery policy and safety critic training
            self.num_unsafe_transitions = 0
            for transition in self.constraint_demo_data:
                self.recovery_memory.push(*transition)
                self.num_constraint_violations += int(transition[2])
                self.num_unsafe_transitions += 1
                if self.num_unsafe_transitions == self.exp_cfg.num_unsafe_transitions:
                    break
            print("Number of Constraint Transitions: ",
                  self.num_unsafe_transitions)
            print("Number of Constraint Violations: ",
                  self.num_constraint_violations)
            # Pass encoder to safety critic:
            self.agent.safety_critic.encoder = self.recovery_policy.get_encoding
            # Train safety critic on encoded states
            for i in range(self.exp_cfg.critic_safe_pretraining_steps):
                if i % 100 == 0:
                    print("CRITIC SAFE UPDATE STEP: ", i)
                self.agent.safety_critic.update_parameters(
                    memory=self.recovery_memory,
                    policy=self.agent.policy,
                    batch_size=min(self.exp_cfg.batch_size,
                                   len(self.constraint_demo_data)))

    def pretrain_task_critic(self):
        self.num_task_transitions = 0
        for transition in self.task_demo_data:
            self.memory.push(*transition)
            self.num_task_transitions += 1
            if self.num_task_transitions == self.exp_cfg.num_task_transitions:
                break
        print("Number of Task Transitions: ", self.num_task_transitions)
        # Pre-train task critic
        for i in range(self.exp_cfg.critic_pretraining_steps):
            if i % 100 == 0:
                print("Update: ", i)
            self.agent.update_parameters(
                self.memory,
                min(self.exp_cfg.batch_size, self.num_task_transitions),
                self.updates,
                safety_critic=self.agent.safety_critic)
            self.updates += 1

    def run(self):
        # Train recovery policy and associated value function on demos
        if not self.exp_cfg.disable_offline_updates and (
                self.exp_cfg.use_recovery or self.exp_cfg.DGD_constraints
                or self.exp_cfg.RCPO):
            self.pretrain_critic_recovery()
        # Optionally initialize task policy with demos
        if self.task_demos:
            self.pretrain_task_critic()

        # Training Loop
        train_rollouts = []
        test_rollouts = []
        for i_episode in itertools.count(1):
            train_rollout_info = self.get_train_rollout(i_episode)
            train_rollouts.append(train_rollout_info)
            if i_episode % 10 == 0 and self.exp_cfg.eval:
                test_rollout_info = self.get_test_rollout(i_episode)
                test_rollouts.append(test_rollout_info)
            if self.total_numsteps > self.exp_cfg.num_steps or i_episode > self.exp_cfg.num_eps:
                break
            self.dump_logs(train_rollouts, test_rollouts)

    def get_train_rollout(self, i_episode):
        episode_reward = 0
        episode_steps = 0
        done = False
        state, info = self.env.reset()
        if self.exp_cfg.cnn:
            state = process_obs(state, self.exp_cfg.env_name)

        train_rollout_info = []
        ep_states = [state]
        ep_actions = []
        ep_constraints = []

        if i_episode % 10 == 0:
            print("SEED: ", self.exp_cfg.seed)
            print("LOGDIR: ", self.logdir)

        while not done:
            if len(self.memory) > self.exp_cfg.batch_size:
                # Number of updates per step in environment
                for i in range(self.exp_cfg.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(
                        self.memory,
                        min(self.exp_cfg.batch_size, len(self.memory)),
                        self.updates,
                        safety_critic=self.agent.safety_critic,
                        nu=self.nu_schedule(i_episode))
                    if not self.exp_cfg.disable_online_updates and len(
                            self.recovery_memory) > self.exp_cfg.batch_size and (
                            self.num_viols + self.num_constraint_violations
                    ) / self.exp_cfg.batch_size > self.exp_cfg.pos_fraction:
                        self.agent.safety_critic.update_parameters(
                            memory=self.recovery_memory,
                            policy=self.agent.policy,
                            batch_size=self.exp_cfg.batch_size,
                            plot=0)
                    self.updates += 1

            # Get action, execute action, and compile step results
            action, real_action, recovery_used = self.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(real_action)
            if terminated or truncated:
                done = True
            info['recovery'] = recovery_used

            if self.exp_cfg.cnn:
                next_state = process_obs(next_state, self.exp_cfg.env_name)

            train_rollout_info.append(info)
            episode_steps += 1
            episode_reward += reward
            self.total_numsteps += 1

            if info['cost']:
                reward -= self.exp_cfg.constraint_reward_penalty

            mask = float(not done)
            done = done or episode_steps == self.env.max_episode_steps

            # Update buffers
            if not self.exp_cfg.disable_action_relabeling:
                self.memory.push(state, action, reward, next_state, mask)
            else:
                self.memory.push(state, real_action, reward, next_state, mask)

            if self.exp_cfg.use_recovery or self.exp_cfg.DGD_constraints or self.exp_cfg.RCPO:
                self.recovery_memory.push(state, real_action,
                                          info['cost'], next_state, mask)
                if recovery_used and self.exp_cfg.add_both_transitions:
                    self.memory.push(state, real_action, reward, next_state,
                                     mask)
            state = next_state
            ep_states.append(state)
            ep_actions.append(real_action)
            ep_constraints.append([info['cost']])

        # Get success/violation stats
        if info['cost']:
            self.num_viols += 1
            if info['recovery']:
                self.viol_and_recovery += 1
            else:
                self.viol_and_no_recovery += 1
        self.num_successes += int(info['is_success'])

        # Update recovery policy using online data
        if self.exp_cfg.use_recovery and not self.exp_cfg.disable_online_updates:
            self.all_ep_data.append({
                'obs': np.array(ep_states),
                'ac': np.array(ep_actions),
                'constraint': np.array(ep_constraints)
            })
            if i_episode % self.exp_cfg.recovery_policy_update_freq == 0 and not (
                    self.exp_cfg.MF_recovery
                    or self.exp_cfg.Q_sampling_recovery
                    or self.exp_cfg.DGD_constraints):
                if not self.exp_cfg.vismpc_recovery:
                    self.train_MB_recovery(
                        [ep_data['obs'] for ep_data in self.all_ep_data],
                        [ep_data['ac'] for ep_data in self.all_ep_data])
                    self.all_ep_data = []
                else:
                    self.recovery_policy.train_online(i_episode)

        # Print performance stats
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".
              format(i_episode, self.total_numsteps, episode_steps,
                     round(episode_reward, 2)))

        print("Num Violations So Far: %d" % self.num_viols)
        print("Violations with Recovery: %d" % self.viol_and_recovery)
        print("Violations with No Recovery: %d" % self.viol_and_no_recovery)
        print("Num Successes So Far: %d" % self.num_successes)
        return train_rollout_info

    def get_test_rollout(self, i_episode):
        avg_reward = 0.
        test_rollout_info = []
        state, info = self.env.reset()

        if 'maze' in self.exp_cfg.env_name:
            im_list = [self.env._get_obs(images=True)]
        elif 'extraction' in self.exp_cfg.env_name:
            im_list = [self.env.render().squeeze()]

        if self.exp_cfg.cnn:
            state = process_obs(state, self.exp_cfg.env_name)

        episode_reward = 0
        episode_steps = 0
        done = False
        while not done:
            action, real_action, recovery_used = self.get_action(state,
                                                                 train=False)
            next_state, reward, terminated, truncated, info = self.env.step(real_action)  # Step
            if terminated or truncated:
                done = True
            info['recovery'] = recovery_used
            done = done or episode_steps == self.env.max_episode_steps

            if 'maze' in self.exp_cfg.env_name:
                im_list.append(self.env._get_obs(images=True))
            elif 'extraction' in self.exp_cfg.env_name:
                im_list.append(self.env.render().squeeze())

            if self.exp_cfg.cnn:
                next_state = process_obs(next_state, self.exp_cfg.env_name)

            test_rollout_info.append(info)
            episode_reward += reward
            episode_steps += 1
            state = next_state

        avg_reward += episode_reward

        if 'maze' in self.exp_cfg.env_name or 'extraction' in self.exp_cfg.env_name:
            npy_to_gif(im_list, osp.join(self.logdir,
                                         "test_" + str(i_episode)))

        print("----------------------------------------")
        print("Avg. Reward: {}".format(round(avg_reward, 2)))
        print("----------------------------------------")
        return test_rollout_info

    def dump_logs(self, train_rollouts, test_rollouts):
        data = {"test_stats": test_rollouts, "train_stats": train_rollouts}
        with open(osp.join(self.logdir, "run_stats.pkl"), "wb") as f:
            pickle.dump(data, f)


    def get_action(self, state, train=True):
        def recovery_thresh(state, action):
            if not self.exp_cfg.use_recovery:
                return False

            critic_val = self.agent.safety_critic.get_value(
                torchify(state).unsqueeze(0),
                torchify(action).unsqueeze(0))

            if critic_val > self.exp_cfg.eps_safe:
                return True
            return False

        if self.exp_cfg.start_steps > self.total_numsteps and train:
            action = self.env.action_space.sample()  # Sample random action
        elif train:
            action = self.agent.select_action(
                state)  # Sample action from policy
        else:
            action = self.agent.select_action(
                state, eval=True)  # Sample action from policy

        if recovery_thresh(state, action):
            recovery = True
            if self.exp_cfg.MF_recovery or self.exp_cfg.Q_sampling_recovery:
                real_action = self.agent.safety_critic.select_action(state)
            else:
                real_action = self.recovery_policy.act(state, 0)
        else:
            recovery = False
            real_action = np.copy(action)
        return action, real_action, recovery