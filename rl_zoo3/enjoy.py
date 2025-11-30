import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path


def enjoy() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    parser.add_argument(
        "--value-stats-path",
        type=str,
        default="",
        help="Path to a .pt file where observations, value predictions, and realized returns will be stored",
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        # Special case for rl-trained agents
        # auto-download from the hub
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=args.exp_id,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=maybe_stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
        vec_env_cls=ExperimentManager.default_vec_env_cls,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            # load models with different obs bounds
            # Note: doesn't work with channel last envs
            # "observation_space": env.observation_space,
        }

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=args.device, **kwargs)
    # Uncomment to save patched file (for instance gym -> gymnasium)
    # model.save(model_path)
    # Patch VecNormalize (gym -> gymnasium)
    # from pathlib import Path
    # env.observation_space = model.observation_space
    # env.action_space = model.action_space
    # env.save(Path(model_path).parent / env_name / "vecnormalize.pkl")

    obs = env.reset()

    def _copy_obs(batch_obs, index):
        if isinstance(batch_obs, dict):
            return {key: np.array(value[index]) for key, value in batch_obs.items()}
        if isinstance(batch_obs, np.ndarray):
            return np.array(batch_obs[index])
        return batch_obs[index]

    track_values = args.value_stats_path != ""
    value_traces = None
    value_records = []
    episode_counters = None
    truncated_episodes = 0
    truncated_steps = 0
    if track_values:
        value_traces = [dict(observations=[], values=[], rewards=[]) for _ in range(env.num_envs)]
        episode_counters = [0 for _ in range(env.num_envs)]

    def flush_value_trace(env_idx: int, episode_done: bool) -> None:
        nonlocal truncated_episodes, truncated_steps
        if not track_values or value_traces is None or episode_counters is None:
            return
        traces = value_traces[env_idx]
        length = min(len(traces["observations"]), len(traces["values"]), len(traces["rewards"]))
        if length == 0:
            value_traces[env_idx] = dict(observations=[], values=[], rewards=[])
            if episode_done:
                episode_counters[env_idx] += 1
            return
        rewards = np.array(traces["rewards"][:length], dtype=float)
        returns = np.cumsum(rewards[::-1])[::-1]
        for step_idx, (obs_snapshot, value_pred, ret) in enumerate(
            zip(traces["observations"][:length], traces["values"][:length], returns)
        ):
            value_records.append(
                dict(
                    env_index=env_idx,
                    episode_index=episode_counters[env_idx],
                    timestep=step_idx,
                    observation=obs_snapshot,
                    value_prediction=float(value_pred),
                    returns=float(ret),
                    episode_done=episode_done,
                )
            )
        value_traces[env_idx] = dict(observations=[], values=[], rewards=[])
        if episode_done:
            episode_counters[env_idx] += 1
        else:
            truncated_episodes += 1
            truncated_steps += length

    # Deterministic by default except for atari games
    stochastic = args.stochastic or ((is_atari or is_minigrid) and not args.deterministic)
    deterministic = not stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    generator = range(args.n_timesteps)
    if args.progress:
        if tqdm is None:
            raise ImportError("Please install tqdm and rich to use the progress bar")
        generator = tqdm(generator)

    try:
        for _ in generator:
            if track_values:
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                predicted_values = (
                    model.policy.predict_values(obs_tensor).detach().cpu().numpy().reshape(env.num_envs)
                )
                for env_idx in range(env.num_envs):
                    value_traces[env_idx]["observations"].append(_copy_obs(obs, env_idx))
                    value_traces[env_idx]["values"].append(float(predicted_values[env_idx]))

            action, lstm_states = model.predict(
                obs,  # type: ignore[arg-type]
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            obs, reward, done, infos = env.step(action)

            if track_values:
                for env_idx in range(env.num_envs):
                    value_traces[env_idx]["rewards"].append(float(reward[env_idx]))

            episode_start = done

            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if args.n_envs == 1:
                # For atari the return reward is not the atari score
                # so we have to get it from the infos dict
                if is_atari and infos is not None and args.verbose >= 1:
                    episode_infos = infos[0].get("episode")
                    if episode_infos is not None:
                        print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                        print("Atari Episode Length", episode_infos["l"])

                if done and not is_atari and args.verbose > 0:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0

                # Reset also when the goal is achieved when using HER
                if done and infos[0].get("is_success") is not None:
                    if args.verbose > 1:
                        print("Success?", infos[0].get("is_success", False))

                    if infos[0].get("is_success") is not None:
                        successes.append(infos[0].get("is_success", False))
                        episode_reward, ep_len = 0.0, 0

            if track_values:
                for env_idx in range(env.num_envs):
                    if done[env_idx]:
                        flush_value_trace(env_idx, episode_done=True)

    except KeyboardInterrupt:
        pass

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    if track_values:
        if value_traces is not None:
            for env_idx in range(env.num_envs):
                flush_value_trace(env_idx, episode_done=False)
        save_path = args.value_stats_path
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        payload = dict(
            records=value_records,
            metadata=dict(
                algo=algo,
                env_id=env_name.gym_id,
                n_timesteps=args.n_timesteps,
                n_envs=env.num_envs,
                truncated_episodes=truncated_episodes,
                truncated_steps=truncated_steps,
                total_records=len(value_records),
            ),
        )
        th.save(payload, save_path)
        if args.verbose > 0:
            print(f"Saved {len(value_records)} value records to {save_path}")
            if truncated_steps > 0:
                print(
                    f"Included {truncated_steps} steps from {truncated_episodes} unfinished episodes when saving value stats"
                )

    env.close()


if __name__ == "__main__":
    enjoy()
