from collections import deque

import numpy as np
import torch

from src.agent import Agent


def ddpg(agent: Agent, env, brain_name, n_agents, n_episodes: int = 10):
    scores_window = deque(maxlen=100)
    scores_mean_agent = []
    scores_mean = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset()[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(n_agents)
        while True:

            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        score = np.mean(scores)
        scores_window.append(score)
        scores_mean_agent.append(score)
        scores_mean.append(np.mean(scores_window))

        print(
            f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}",
            end="",
        )
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")
        if np.mean(scores_window) >= 30.0:
            print(
                f"\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score:"
                f" {np.mean(scores_window):.2f}"
            )
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
            break

        if np.mean(scores_window) >= 30.0:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_window)
                )
            )
            torch.save(agent.policy_network_local.state_dict(), "checkpoint_policy.pth")
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint_qnetwork.pth")
            print("saved networks")
            break
    return scores_mean_agent, scores_mean
