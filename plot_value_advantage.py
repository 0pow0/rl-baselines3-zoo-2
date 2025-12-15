import argparse
import importlib
import os
from typing import Any, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import yaml
from gymnasium import spaces
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.utils import obs_as_tensor, set_random_seed
from stable_baselines3.common.vec_env import VecEnv, VecTransposeImage, is_vecenv_wrapped

import rl_zoo3.import_envs  # noqa: F401
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot correlation between value predictions and returns")
    parser.add_argument("--env", type=EnvironmentName, default="CartPole-v1", help="Environment ID")
    parser.add_argument("-f", "--folder", type=str, default="rl-trained-agents", help="Log folder")
    parser.add_argument("--algo", type=str, default="ppo", choices=list(ALGOS.keys()), help="RL algorithm")
    parser.add_argument("--exp-id", type=int, default=0, help="Experiment ID (0: latest, -1: no exp folder)")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of vectorized environments")
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=0,
        help="Number of steps to collect per environment (default: use model's n_steps)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device (cpu, cuda, auto)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-threads", type=int, default=-1, help="PyTorch thread count (-1 keeps default)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: silent, 1: info)")
    parser.add_argument("--norm-reward", action="store_true", help="Use VecNormalize statistics if available")
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional Gym packages to import so environments register",
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional keyword arguments for the environment constructor",
    )
    parser.add_argument(
        "--load-best", action="store_true", help="Load best model instead of last model if checkpoints exist"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load a specific checkpoint (timesteps) instead of the final model if available",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        help="Load the latest checkpoint instead of the final model if available",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="value_return_correlation.png",
        help="Path where the scatter plot will be saved",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Only save the figure without trying to display it (useful on headless setups)",
    )
    return parser.parse_args()


def maybe_transpose_env(env: VecEnv, verbose: int = 0) -> VecEnv:
    if is_vecenv_wrapped(env, VecTransposeImage):
        return env
    wrap = False
    obs_space = env.observation_space
    if isinstance(obs_space, spaces.Dict):
        for space in obs_space.spaces.values():
            if is_image_space(space) and not is_image_space_channels_first(space):  # type: ignore[arg-type]
                wrap = True
                break
    else:
        wrap = is_image_space(obs_space) and not is_image_space_channels_first(obs_space)  # type: ignore[arg-type]
    if wrap:
        if verbose > 0:
            print("Wrapping the environment with VecTransposeImage to convert channel order.")
        env = VecTransposeImage(env)
    return env


def load_model_and_env(args: argparse.Namespace) -> Tuple[EnvironmentName, str, str, th.nn.Module, VecEnv]:
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_name: EnvironmentName = args.env

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            args.folder,
            args.algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as exc:
        print("Model not found locally, attempting to download from the SB3 HuggingFace hub...")
        download_from_hub(
            algo=args.algo,
            env_name=env_name,
            exp_id=args.exp_id,
            folder=args.folder,
            organization="sb3",
            repo_name=None,
            force=False,
        )
        _, model_path, log_path = get_model_path(
            args.exp_id,
            args.folder,
            args.algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )

    if args.verbose > 0:
        print(f"Loading model from {model_path}")

    set_random_seed(args.seed)
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as fh:
            loaded_args = yaml.load(fh, Loader=yaml.UnsafeLoader)
            if loaded_args.get("env_kwargs") is not None:
                env_kwargs = loaded_args["env_kwargs"]
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    env: VecEnv = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=maybe_stats_path,
        seed=args.seed,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
        should_render=False,
        vec_env_cls=ExperimentManager.default_vec_env_cls,
    )
    env = maybe_transpose_env(env, verbose=args.verbose)

    model_kwargs: dict[str, Any] = dict(seed=args.seed)
    model = ALGOS[args.algo].load(model_path, device=args.device, **model_kwargs)

    return env_name, model_path, log_path, model, env


def collect_buffer(model, env, rollout_steps: int) -> RolloutBuffer:
    gamma = getattr(model, "gamma", 0.99)
    gae_lambda = getattr(model, "gae_lambda", 1.0)
    print(f"{gamma=} {gae_lambda=}")
    buffer = RolloutBuffer(
        rollout_steps,
        env.observation_space,  # type: ignore[arg-type]
        env.action_space,
        device=model.device,
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=env.num_envs,
    )
    obs = env.reset()
    episode_starts = np.ones((env.num_envs,), dtype=np.float32)
    steps_collected = 0

    while steps_collected < rollout_steps:
        obs_tensor = obs_as_tensor(obs, model.device)
        with th.no_grad():
            actions, values, log_probs = model.policy(obs_tensor)
        actions_np = actions.cpu().numpy()
        clipped_actions = actions_np
        if isinstance(env.action_space, spaces.Box):
            if model.policy.squash_output:
                clipped_actions = model.policy.unscale_action(clipped_actions)
            else:
                clipped_actions = np.clip(actions_np, env.action_space.low, env.action_space.high)

        new_obs, rewards, dones, infos = env.step(clipped_actions)
        rewards = np.array(rewards)

        for idx in range(env.num_envs):
            if (
                dones[idx]
                and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = model.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = model.policy.predict_values(terminal_obs)[0]
                rewards[idx] += gamma * float(terminal_value)

        buffer.add(obs, actions_np, rewards, episode_starts, values, log_probs)
        obs = new_obs
        episode_starts = dones
        steps_collected += 1

    with th.no_grad():
        last_values = model.policy.predict_values(obs_as_tensor(obs, model.device))
    buffer.compute_returns_and_advantage(last_values=last_values, dones=episode_starts)
    return buffer


def plot_correlation(values: np.ndarray, returns: np.ndarray, save_path: str, title: str, show: bool) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(values, returns, s=10, alpha=0.35, edgecolors="none")
    lim_min = float(np.min((values.min(), returns.min())))
    lim_max = float(np.max((values.max(), returns.max())))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], color="tab:red", linewidth=1.2, linestyle="--", label="Ideal")
    plt.xlabel("Value Prediction")
    plt.ylabel("Returns")
    plt.title(title)
    plt.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    plt.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    args = parse_args()
    env_name, _, _, model, env = load_model_and_env(args)
    rollout_steps = args.rollout_steps if args.rollout_steps > 0 else getattr(model, "n_steps", 2048)

    if args.verbose > 0:
        print(f"Collecting {rollout_steps} steps per environment ({rollout_steps * env.num_envs} transitions total)")

    buffer = collect_buffer(model, env, rollout_steps)
    env.close()

    values = buffer.values.reshape(-1)
    returns = buffer.returns.reshape(-1)
    mask = np.isfinite(values) & np.isfinite(returns)
    values = values[mask]
    returns = returns[mask]
    if values.size == 0:
        raise RuntimeError("No valid samples were collected to compute the correlation.")

    denominator = np.std(values) * np.std(returns)
    correlation = 0.0
    if denominator > 0 and values.size > 1:
        correlation = float(np.corrcoef(values, returns)[0, 1])

    if args.verbose > 0:
        print(f"Correlation between value predictions and returns: {correlation:.4f}")
        print(f"Saving scatter plot to {args.output}")

    title = f"{env_name.gym_id} / {args.algo.upper()} (rho={correlation:.3f})"
    plot_correlation(values, returns, args.output, title, show=not args.no_show)


if __name__ == "__main__":
    main()
