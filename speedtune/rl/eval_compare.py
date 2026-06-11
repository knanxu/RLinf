"""Side-by-side eval: trained Rainbow agent vs default baseline (v=1, vs=1, as=1).

跑 N 个 episode (相同 seed 序列) 比较 trained 与 baseline 的成功率和 wall-time.
第一个 episode 把两边的 head_camera 帧序列拼成左右并排 mp4.

帧采集: monkey-patch task_env.scene.step, 每 K 步调一次
        task_env.cameras.get_rgb()['head_camera'] 拿一帧, 比 get_obs() 便宜.

种子对齐: 两边按相同 seed 列表分别 reset, 物体初始 pose 一致.

用法 (从 RLinf 仓根运行; RoboTwin sim 经 ROBOTWIN_PATH 接入):
    cd $WORKSPACE/RLinf
    export ROBOTWIN_PATH=$WORKSPACE/RoboTwin
    export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH
    python -m speedtune.rl.eval_compare \
        --ckpt speedtune/rl/runs/rainbow_chunk_speedup_<ts>/rainbow_step20000.pt \
        --task_name shake_bottle --task_config demo_clean \
        --server_host 127.0.0.1 --server_port 8000 \
        --n_episodes 10 \
        --output_dir speedtune/rl/eval_compare_runs
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Callable, List, Optional

import numpy as np

from .env import ChunkSpeedupEnv
from .rainbow.agent import RainbowAgent
from .rainbow.config import FullConfig as RainbowFullConfig
from .config import (
    ActionSpaceConfig as _SACActionSpaceConfig,
    EnvConfig as _SACEnvConfig,
    PolicyConfig as _SACPolicyConfig,
    RewardConfig as _SACRewardConfig,
)


# -----------------------------------------------------------------------
# Adapter helpers (复用 train.py 里的转换, 这里复制一份保持脚本独立)
# -----------------------------------------------------------------------
def _to_sac_action_space(cfg: RainbowFullConfig) -> _SACActionSpaceConfig:
    g = cfg.action_grid
    return _SACActionSpaceConfig(
        v_low=float(g.v_grid[0]), v_high=float(g.v_grid[-1]),
        vel_low=float(g.vel_grid[0]), vel_high=float(g.vel_grid[-1]),
        acc_low=float(g.acc_grid[0]), acc_high=float(g.acc_grid[-1]),
    )


def _to_sac_reward(cfg: RainbowFullConfig) -> _SACRewardConfig:
    r = cfg.reward
    return _SACRewardConfig(
        alpha_v=r.alpha_v, alpha_vs=r.alpha_vs, alpha_as=r.alpha_as,
        beta_v=r.beta_v, beta_vs=r.beta_vs, beta_as=r.beta_as,
        fallback_penalty=r.fallback_penalty, crash_penalty=r.crash_penalty,
    )


def _to_sac_env(cfg: RainbowFullConfig) -> _SACEnvConfig:
    e = cfg.env
    return _SACEnvConfig(
        task_name=e.task_name, task_config=e.task_config,
        instruction_type=e.instruction_type,
        chunk_size=e.chunk_size, max_chunks_per_episode=e.max_chunks_per_episode,
        cond_emb_dim=e.cond_emb_dim, state_dim=e.state_dim,
    )


def _to_sac_policy(cfg: RainbowFullConfig) -> _SACPolicyConfig:
    p = cfg.policy
    return _SACPolicyConfig(
        server_host=p.server_host, server_port=p.server_port,
        api_key=p.api_key, pi0_step=p.pi0_step,
    )


# -----------------------------------------------------------------------
# Frame recorder: monkey-patch task_env.scene.step
# -----------------------------------------------------------------------
class FrameRecorder:
    """每 sample_every 个 scene.step 抓一帧 head_camera RGB.

    顺便累计 ``total_record_time_s``: 每帧 _update_render + take_picture +
    get_rgb 的耗时总和. 这部分时间是评估器人为加进去的, 不属于实机部署
    的真实执行时间, eval runner 会从 wall_time_no_obs 里再扣一次, 得到
    "真实部署执行时间".
    """

    def __init__(self, sample_every: int = 10):
        self.sample_every = max(1, int(sample_every))
        self.frames: List[np.ndarray] = []
        self._counter = 0
        self._installed_env = None
        self._orig_step = None
        self.total_record_time_s: float = 0.0

    def attach(self, task_env) -> None:
        if self._installed_env is not None:
            self.detach()
        scene = task_env.scene
        cameras = task_env.cameras
        recorder = self

        orig_step = scene.step

        def patched_step(*args, **kwargs):
            ret = orig_step(*args, **kwargs)
            recorder._counter += 1
            if recorder._counter % recorder.sample_every == 0:
                _t0 = time.perf_counter()
                try:
                    task_env._update_render()
                    cameras.update_picture()
                    rgb_dict = cameras.get_rgb()
                    rgb = rgb_dict["head_camera"]["rgb"]
                    if rgb.dtype != np.uint8:
                        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                    recorder.frames.append(np.ascontiguousarray(rgb))
                except Exception:
                    pass
                recorder.total_record_time_s += time.perf_counter() - _t0
            return ret

        scene.step = patched_step
        self._orig_step = orig_step
        self._installed_env = task_env

    def detach(self) -> None:
        if self._installed_env is not None and self._orig_step is not None:
            try:
                self._installed_env.scene.step = self._orig_step
            except Exception:
                pass
        self._installed_env = None
        self._orig_step = None

    def reset_buffer(self) -> None:
        self.frames = []
        self._counter = 0
        self.total_record_time_s = 0.0


# -----------------------------------------------------------------------
# Episode runner
# -----------------------------------------------------------------------
def _run_episode(
    env: ChunkSpeedupEnv,
    seed: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
    recorder: Optional[FrameRecorder] = None,
) -> dict:
    """跑一个 episode, action 由 action_fn(obs)->np.ndarray[3] 决定."""
    obs, _ = env.reset(seed=seed)
    if recorder is not None:
        recorder.reset_buffer()
        recorder.attach(env._task_env)

    done = False
    ep_return = 0.0
    n_chunks = 0
    last_info: dict = {}
    while not done:
        action = action_fn(obs)
        obs, r, term, trunc, info = env.step(action)
        ep_return += float(r)
        n_chunks += 1
        last_info = info
        done = term or trunc

    if recorder is not None:
        recorder.detach()

    record_overhead_s = float(recorder.total_record_time_s) if recorder is not None else 0.0
    wall_no_obs = float(last_info.get("episode_wall_time_no_obs", 0.0))
    pi_infer_s = float(last_info.get("episode_pi_infer_time", 0.0))
    # 真实硬件执行时间 = dense_steps / 250Hz, 累加每个 chunk.
    # 不受仿真器快慢 / 评估器额外渲染影响, 是实机部署上动作真正消耗的时间.
    real_exec_s = float(last_info.get("episode_real_exec_time", 0.0))
    deploy_time_s = pi_infer_s + real_exec_s

    return {
        "seed": int(seed),
        "success": bool(last_info.get("success", False)),
        "episode_return": float(ep_return),
        "n_chunks": int(n_chunks),
        "wall_time_s": float(last_info.get("episode_wall_time", 0.0)),
        "wall_time_no_obs_s": wall_no_obs,
        "frame_record_overhead_s": record_overhead_s,
        "pi_infer_time_s": pi_infer_s,
        "real_exec_time_s": real_exec_s,
        "deploy_time_s": deploy_time_s,
        "topp_total_s": float(last_info.get("episode_total_topp_time", 0.0)),
    }


# -----------------------------------------------------------------------
# Video composition (cv2 hconcat with text overlay)
# -----------------------------------------------------------------------
def _pad_to_length(frames: List[np.ndarray], target_len: int) -> List[np.ndarray]:
    if not frames:
        return frames
    if len(frames) >= target_len:
        return frames[:target_len]
    last = frames[-1]
    return frames + [last] * (target_len - len(frames))


def _put_text(img: np.ndarray, lines: List[str], origin=(10, 30)) -> np.ndarray:
    import cv2
    out = img.copy()
    x, y = origin
    for i, line in enumerate(lines):
        yi = y + i * 28
        cv2.putText(out, line, (x + 1, yi + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (x, yi), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _make_summary_frame(width: int, height: int, lines: List[str]) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    return _put_text(img, lines, origin=(20, 60))


def write_compare_video(
    trained_frames: List[np.ndarray],
    baseline_frames: List[np.ndarray],
    trained_summary: dict,
    baseline_summary: dict,
    out_path: str,
    fps: int = 25,
) -> None:
    """把两边帧序列左右并排写成一个 mp4. 短的一侧最后一帧 pad."""
    import cv2

    if not trained_frames or not baseline_frames:
        print(f"[video] empty frames (trained={len(trained_frames)} "
              f"baseline={len(baseline_frames)}), skip", flush=True)
        return

    h_t, w_t, _ = trained_frames[0].shape
    h_b, w_b, _ = baseline_frames[0].shape
    H = max(h_t, h_b)
    W = w_t + w_b
    label_band = 40

    def _resize_to_h(frame, target_h):
        if frame.shape[0] == target_h:
            return frame
        scale = target_h / frame.shape[0]
        new_w = int(round(frame.shape[1] * scale))
        return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

    target_len = max(len(trained_frames), len(baseline_frames))
    t_frames = _pad_to_length(trained_frames, target_len)
    b_frames = _pad_to_length(baseline_frames, target_len)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_h = H + label_band
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")

    label_left = "Trained (Rainbow)"
    label_right = "Baseline (v=1, vs=1, as=1)"

    for i in range(target_len):
        ft = _resize_to_h(t_frames[i], H)
        fb = _resize_to_h(b_frames[i], H)
        if ft.shape[1] + fb.shape[1] != W:
            W_actual = ft.shape[1] + fb.shape[1]
            row = np.zeros((H, W_actual, 3), dtype=np.uint8)
        else:
            W_actual = W
            row = np.zeros((H, W_actual, 3), dtype=np.uint8)
        row[:, : ft.shape[1]] = ft
        row[:, ft.shape[1]:] = fb
        # head_camera RGB -> BGR for cv2
        row_bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)

        band = np.full((label_band, W_actual, 3), 30, dtype=np.uint8)
        band = _put_text(band, [label_left], origin=(10, 28))
        band = _put_text(
            band, [label_right],
            origin=(ft.shape[1] + 10, 28),
        )
        frame = np.vstack([band, row_bgr])
        if frame.shape[1] != W:
            frame = cv2.resize(frame, (W, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)

    summary_lines_left = [
        "Trained (Rainbow)",
        f"success: {trained_summary['success']}",
        f"deploy_time: {trained_summary['deploy_time_s']:.2f}s",
        f"  pi_infer:  {trained_summary['pi_infer_time_s']:.2f}s",
        f"  real_exec: {trained_summary['real_exec_time_s']:.2f}s",
        f"chunks: {trained_summary['n_chunks']}",
    ]
    summary_lines_right = [
        "Baseline",
        f"success: {baseline_summary['success']}",
        f"deploy_time: {baseline_summary['deploy_time_s']:.2f}s",
        f"  pi_infer:  {baseline_summary['pi_infer_time_s']:.2f}s",
        f"  real_exec: {baseline_summary['real_exec_time_s']:.2f}s",
        f"chunks: {baseline_summary['n_chunks']}",
    ]
    summary_left = _make_summary_frame(W // 2, H, summary_lines_left)
    summary_right = _make_summary_frame(W - W // 2, H, summary_lines_right)
    summary_row = np.hstack([summary_left, summary_right])
    summary_row_bgr = cv2.cvtColor(summary_row, cv2.COLOR_RGB2BGR)
    band = np.full((label_band, W, 3), 30, dtype=np.uint8)
    band = _put_text(band, ["SUMMARY"], origin=(10, 28))
    summary_frame = np.vstack([band, summary_row_bgr])
    for _ in range(fps * 3):
        writer.write(summary_frame)

    writer.release()
    print(f"[video] wrote {out_path} ({target_len + fps * 3} frames @ {fps}fps)", flush=True)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def _build_agent(ckpt_path: str, cfg: RainbowFullConfig) -> RainbowAgent:
    grids = cfg.action_grid
    K_tuple = (len(grids.v_grid), len(grids.vel_grid), len(grids.acc_grid))
    agent = RainbowAgent(
        state_dim=cfg.env.state_dim,
        num_actions_per_dim=K_tuple,
        action_grids=(
            np.asarray(grids.v_grid, dtype=np.float32),
            np.asarray(grids.vel_grid, dtype=np.float32),
            np.asarray(grids.acc_grid, dtype=np.float32),
        ),
        c51_cfg=cfg.c51,
        rainbow_cfg=cfg.rainbow,
        net_cfg=cfg.network,
        device=cfg.train.device,
    )
    agent.load(ckpt_path)
    return agent


def _override_cfg(cfg: RainbowFullConfig, args: argparse.Namespace) -> RainbowFullConfig:
    flat = {}
    for sub in (cfg.action_grid, cfg.reward, cfg.env, cfg.policy,
                cfg.network, cfg.c51, cfg.rainbow, cfg.train):
        flat.update({k: (sub, k) for k in asdict(sub).keys()})
    for k, v in vars(args).items():
        if v is None:
            continue
        if k in flat:
            sub, key = flat[k]
            cur = getattr(sub, key)
            try:
                cast = type(cur)(v) if cur is not None else v
            except Exception:
                cast = v
            setattr(sub, key, cast)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to rainbow_stepN.pt checkpoint.")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--server_host", type=str, default=None)
    parser.add_argument("--server_port", type=int, default=None)
    parser.add_argument("--pi0_step", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Episodes per side (trained / baseline).")
    parser.add_argument("--seed_base", type=int, default=20260101,
                        help="seed_i = seed_base + i for both sides.")
    parser.add_argument("--sample_every", type=int, default=10,
                        help="Sample one head_camera frame every N scene.step()s. "
                             "scene runs at 250Hz, default 10 -> 25 fps video.")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--output_dir", type=str,
                        default="speedtune/rl/eval_compare_runs")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Subdir name. Defaults to compare_<unix_ts>.")
    args = parser.parse_args()

    cfg = RainbowFullConfig()
    cfg = _override_cfg(cfg, args)

    run_name = args.run_name or f"compare_{int(time.time())}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[eval] run dir: {run_dir}", flush=True)

    env = ChunkSpeedupEnv(
        env_cfg=_to_sac_env(cfg),
        action_cfg=_to_sac_action_space(cfg),
        reward_cfg=_to_sac_reward(cfg),
        policy_cfg=_to_sac_policy(cfg),
        seed=cfg.train.seed,
        device=cfg.train.device,
    )

    agent = _build_agent(args.ckpt, cfg)
    print(f"[eval] loaded ckpt: {args.ckpt}", flush=True)

    seeds = [args.seed_base + i for i in range(args.n_episodes)]

    def trained_action(obs: np.ndarray) -> np.ndarray:
        a, _idx = agent.select_action(obs, env_step=10**9, deterministic=True)
        return a

    BASELINE = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def baseline_action(_obs: np.ndarray) -> np.ndarray:
        return BASELINE.copy()

    trained_results: List[dict] = []
    baseline_results: List[dict] = []
    trained_frames_first: List[np.ndarray] = []
    baseline_frames_first: List[np.ndarray] = []

    # ----- trained side -----
    print(f"[eval] running TRAINED for {len(seeds)} episodes ...", flush=True)
    for i, seed in enumerate(seeds):
        rec = FrameRecorder(args.sample_every) if i == 0 else None
        info = _run_episode(env, seed, trained_action, recorder=rec)
        trained_results.append(info)
        if rec is not None:
            trained_frames_first = list(rec.frames)
        print(f"  [trained ep{i} seed={seed}] "
              f"success={info['success']} "
              f"deploy={info['deploy_time_s']:.2f}s "
              f"(pi_infer={info['pi_infer_time_s']:.2f}s "
              f"+ real_exec={info['real_exec_time_s']:.2f}s) "
              f"chunks={info['n_chunks']}", flush=True)

    # ----- baseline side -----
    print(f"[eval] running BASELINE for {len(seeds)} episodes ...", flush=True)
    for i, seed in enumerate(seeds):
        rec = FrameRecorder(args.sample_every) if i == 0 else None
        info = _run_episode(env, seed, baseline_action, recorder=rec)
        baseline_results.append(info)
        if rec is not None:
            baseline_frames_first = list(rec.frames)
        print(f"  [baseline ep{i} seed={seed}] "
              f"success={info['success']} "
              f"deploy={info['deploy_time_s']:.2f}s "
              f"(pi_infer={info['pi_infer_time_s']:.2f}s "
              f"+ real_exec={info['real_exec_time_s']:.2f}s) "
              f"chunks={info['n_chunks']}", flush=True)

    env.close()

    # ----- compose video for episode 0 -----
    video_path = os.path.join(run_dir, "compare_ep0.mp4")
    write_compare_video(
        trained_frames_first, baseline_frames_first,
        trained_results[0], baseline_results[0],
        out_path=video_path, fps=args.fps,
    )

    # ----- aggregate metrics -----
    def _agg(rs: List[dict]) -> dict:
        succs = [r["success"] for r in rs]
        succ_deploy = [r["deploy_time_s"] for r in rs if r["success"]]
        succ_pi = [r["pi_infer_time_s"] for r in rs if r["success"]]
        succ_exec = [r["real_exec_time_s"] for r in rs if r["success"]]
        succ_walls = [r["wall_time_no_obs_s"] for r in rs if r["success"]]
        return {
            "n_episodes": len(rs),
            "success_rate": float(np.mean(succs)) if succs else 0.0,
            "n_success": int(sum(succs)),
            "deploy_time_s_mean_success_only":
                float(np.mean(succ_deploy)) if succ_deploy else None,
            "deploy_time_s_min_success_only":
                float(np.min(succ_deploy)) if succ_deploy else None,
            "deploy_time_s_max_success_only":
                float(np.max(succ_deploy)) if succ_deploy else None,
            "pi_infer_time_s_mean_success_only":
                float(np.mean(succ_pi)) if succ_pi else None,
            "real_exec_time_s_mean_success_only":
                float(np.mean(succ_exec)) if succ_exec else None,
            "wall_time_no_obs_s_mean_success_only":
                float(np.mean(succ_walls)) if succ_walls else None,
            "wall_time_no_obs_s_min_success_only":
                float(np.min(succ_walls)) if succ_walls else None,
            "wall_time_no_obs_s_max_success_only":
                float(np.max(succ_walls)) if succ_walls else None,
        }

    summary = {
        "ckpt": os.path.abspath(args.ckpt),
        "task_name": cfg.env.task_name,
        "task_config": cfg.env.task_config,
        "seeds": seeds,
        "trained": {
            "aggregate": _agg(trained_results),
            "episodes": trained_results,
        },
        "baseline": {
            "aggregate": _agg(baseline_results),
            "episodes": baseline_results,
        },
        "video": os.path.abspath(video_path),
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] summary: {summary_path}", flush=True)

    # ----- pretty print -----
    t_agg = summary["trained"]["aggregate"]
    b_agg = summary["baseline"]["aggregate"]
    print("\n========== RESULTS ==========")
    print("  deploy_time_s = pi_infer_time_s + real_exec_time_s")
    print("  pi_infer_time_s : sum of all _infer_pi() wall times (websocket round-trip)")
    print("  real_exec_time_s: sum of dense_steps / 250Hz across all chunks")
    print("                    (true robot-hardware execution time, not sim wall time)")
    print(f"  trained:  sr={t_agg['success_rate']:.2f} "
          f"({t_agg['n_success']}/{t_agg['n_episodes']})  "
          f"deploy={t_agg['deploy_time_s_mean_success_only']}s "
          f"(pi_infer={t_agg['pi_infer_time_s_mean_success_only']}s + "
          f"real_exec={t_agg['real_exec_time_s_mean_success_only']}s)")
    print(f"  baseline: sr={b_agg['success_rate']:.2f} "
          f"({b_agg['n_success']}/{b_agg['n_episodes']})  "
          f"deploy={b_agg['deploy_time_s_mean_success_only']}s "
          f"(pi_infer={b_agg['pi_infer_time_s_mean_success_only']}s + "
          f"real_exec={b_agg['real_exec_time_s_mean_success_only']}s)")
    print(f"  video:    {video_path}")
    print("=============================\n")


if __name__ == "__main__":
    main()
