# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""专家数据回放冒烟测试 (加速可行性, 无 RL) —— 已移植到 RLinf, 从 RLinf 根运行.

从 RoboTwin 采的 hdf5 读 joint_action/vector, 切成 chunk_size 帧的块, 三种模式回放对比:
  A. per_frame:  原版逐帧 take_action (baseline, scale=1.0)
  B. chunk:      take_chunk_action 整段 TOPPRA (不重构 chunk)
  C. reconstruct+chunk:  reconstruct_chunk + take_chunk_action, 多组 (vel/acc/v) scale

打印每档的 success / take_action_cnt / chunk 数 / wall_time(_no_obs) / topp_fail.
**这是投入 RL 前最便宜的可行性检查**: 若加速档 (vs2/as4/v1.5) 的 success 还保持、
wall_time 下降, 说明 TOPPRA 加速物理可行, 再训 Rainbow 才有意义.

需要 RoboTwin sim (经 ROBOTWIN_PATH) + 已采的专家数据 (默认在 ROBOTWIN_PATH/data 下).

Run (从 RLinf 根):
    export ROBOTWIN_PATH=/path/to/RoboTwin
    export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH   # 使 `import envs` (RoboTwin) 解析
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    python speedtune/tests/smoke_replay_expert.py \
        --task_name shake_bottle --task_config smoke_test
"""
import argparse
import importlib
import os
import sys
import time

import h5py
import numpy as np
import yaml

# RoboTwin repo (provides envs.* + task_config/ + expert data). Resolve from
# ROBOTWIN_PATH so this runs from the RLinf root after the port.
_ROBOTWIN = os.environ.get("ROBOTWIN_PATH")
if not _ROBOTWIN:
    raise RuntimeError(
        "set ROBOTWIN_PATH=/path/to/RoboTwin (provides envs.*, task_config/, data/) "
        "and put it on PYTHONPATH so `import envs` resolves."
    )
_ROBOTWIN = os.path.abspath(_ROBOTWIN)
if _ROBOTWIN not in sys.path:
    sys.path.append(_ROBOTWIN)

from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.chunk_accel import reconstruct_chunk  # noqa: E402

CHUNK_SIZE = 50


# ---------- Env setup ----------

def _class_decorator(task_name):
    mod = importlib.import_module(f"envs.{task_name}")
    return getattr(mod, task_name)()


def _build_env_args(task_name: str, task_config: str) -> dict:
    with open(os.path.join(_ROBOTWIN, "task_config", f"{task_config}.yml"), "r") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
        emb_types = yaml.safe_load(f)

    et = args.get("embodiment")
    if len(et) == 1:
        args["left_robot_file"] = emb_types[et[0]]["file_path"]
        args["right_robot_file"] = emb_types[et[0]]["file_path"]
        args["dual_arm_embodied"] = True
    else:
        raise NotImplementedError("only single-embodiment supported in smoke test")

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r") as f:
        cam_cfg = yaml.safe_load(f)
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cam_cfg[head]["h"]
    args["head_camera_w"] = cam_cfg[head]["w"]

    def _get_emb_cfg(file_path: str) -> dict:
        with open(os.path.join(file_path, "config.yml"), "r") as f:
            return yaml.safe_load(f)

    args["left_embodiment_config"] = _get_emb_cfg(args["left_robot_file"])
    args["right_embodiment_config"] = _get_emb_cfg(args["right_robot_file"])
    args["play_once_path_file_list"] = []
    args["eval_mode"] = True       # 触发 step_lim 从 _eval_step_limit.yml 读
    args["collect_data"] = False   # 禁止采集 / 写 cache
    return args


def setup_env(task_env, seed: int, env_args: dict):
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **env_args)


def close_env(task_env):
    try:
        task_env.close_env()
    except Exception as e:
        print(f"[warn] close_env: {e}")


# ---------- Replay modes ----------

def replay_per_frame(task_env, actions: np.ndarray, chunk_size: int = CHUNK_SIZE):
    """Mode A: 按原版逐帧 take_action 回放整个 action 序列."""
    task_env._last_success_obs_time = 0.0
    t0 = time.time()
    topp_fail = 0
    n_frames = 0
    for i in range(0, len(actions), chunk_size):
        chunk = actions[i : i + chunk_size]
        for act in chunk:
            if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success:
                break
            task_env.take_action(act, action_type="qpos", vel_scale=1.0, acc_scale=1.0)
            n_frames += 1
        if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success:
            break
    wall = time.time() - t0
    obs_time = float(getattr(task_env, "_last_success_obs_time", 0.0))
    return {
        "mode": "per_frame",
        "success": bool(task_env.eval_success),
        "take_action_cnt": int(task_env.take_action_cnt),
        "n_chunk_calls": n_frames,
        "topp_fail": topp_fail,
        "wall_time": wall,
        "success_obs_time": obs_time,
        "wall_time_no_obs": max(0.0, wall - obs_time),
    }


def replay_chunk(task_env, actions: np.ndarray, vel_scale: float, acc_scale: float,
                 v_reconstruct: float = 1.0, chunk_size: int = CHUNK_SIZE):
    """Mode B/C: 切块 + (可选重构) + take_chunk_action."""
    t0 = time.time()
    topp_fail = 0
    n_chunks = 0
    obs_time_sum = 0.0
    for i in range(0, len(actions), chunk_size):
        chunk = actions[i : i + chunk_size]
        if chunk.shape[0] < 2:
            break
        if v_reconstruct != 1.0:
            chunk = reconstruct_chunk(chunk, v_reconstruct)
        info = task_env.take_chunk_action(chunk, vel_scale=vel_scale, acc_scale=acc_scale)
        n_chunks += 1
        obs_time_sum += float(info.get("success_obs_time", 0.0))
        if info["status"] == "topp_fallback":
            topp_fail += 1
        if info["status"] == "truncated":
            break
        if task_env.eval_success:
            break
    wall = time.time() - t0
    return {
        "mode": f"chunk(vs={vel_scale},as={acc_scale},v={v_reconstruct})",
        "success": bool(task_env.eval_success),
        "take_action_cnt": int(task_env.take_action_cnt),
        "n_chunk_calls": n_chunks,
        "topp_fail": topp_fail,
        "wall_time": wall,
        "success_obs_time": obs_time_sum,
        "wall_time_no_obs": max(0.0, wall - obs_time_sum),
    }


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="shake_bottle")
    parser.add_argument("--task_config", type=str, default="smoke_test")
    parser.add_argument("--data_root", type=str, default=None,
                        help="expert hdf5 dir; default ROBOTWIN_PATH/data/<task>/<config>/data")
    parser.add_argument("--episodes", type=int, default=-1, help="-1 = all available")
    args = parser.parse_args()

    data_root = args.data_root or os.path.join(
        _ROBOTWIN, "data", args.task_name, args.task_config, "data"
    )
    seed_file = os.path.join(
        _ROBOTWIN, "data", args.task_name, args.task_config, "seed.txt"
    )

    if not os.path.exists(seed_file):
        raise FileNotFoundError(f"seed file missing: {seed_file}")
    with open(seed_file) as f:
        seeds = [int(s) for s in f.read().split()]

    hdf5_files = sorted(
        [f for f in os.listdir(data_root) if f.endswith(".hdf5")],
        key=lambda x: int(x.replace("episode", "").replace(".hdf5", "")),
    )
    if args.episodes > 0:
        hdf5_files = hdf5_files[: args.episodes]
        seeds = seeds[: args.episodes]
    assert len(hdf5_files) == len(seeds), (
        f"hdf5 count {len(hdf5_files)} != seed count {len(seeds)}"
    )

    env_args = _build_env_args(args.task_name, args.task_config)

    configurations = [
        ("per_frame_vs1_as1", "per_frame", {}),
        ("chunk_vs1_as1_v1",   "chunk", {"vel_scale": 1.0, "acc_scale": 1.0, "v_reconstruct": 1.0}),
        ("chunk_vs2_as4_v1",   "chunk", {"vel_scale": 2.0, "acc_scale": 4.0, "v_reconstruct": 1.0}),
        ("chunk_vs1_as1_v1.5", "chunk", {"vel_scale": 1.0, "acc_scale": 1.0, "v_reconstruct": 1.5}),
        ("chunk_vs2_as4_v1.5", "chunk", {"vel_scale": 2.0, "acc_scale": 4.0, "v_reconstruct": 1.5}),
    ]

    all_results = {name: [] for name, _, _ in configurations}

    for cfg_name, mode, kwargs in configurations:
        print(f"\n{'='*78}\n=== Running config: {cfg_name} ===\n{'='*78}")
        task_env = _class_decorator(args.task_name)
        for ep_i, (hdf5_name, seed) in enumerate(zip(hdf5_files, seeds)):
            hdf5_path = os.path.join(data_root, hdf5_name)
            with h5py.File(hdf5_path, "r") as f:
                actions = f["joint_action"]["vector"][...]
            print(f"\n  -- episode {ep_i} (seed={seed}) actions shape {actions.shape} --")

            try:
                setup_env(task_env, seed=seed, env_args=env_args)
            except Exception as e:
                print(f"  [ERR] setup_demo: {e}")
                continue

            try:
                if mode == "per_frame":
                    result = replay_per_frame(task_env, actions)
                else:
                    result = replay_chunk(task_env, actions, **kwargs)
            except Exception as e:
                import traceback
                traceback.print_exc()
                result = {"mode": cfg_name, "success": False, "error": str(e)}
            finally:
                close_env(task_env)

            result["seed"] = seed
            result["episode"] = ep_i
            all_results[cfg_name].append(result)
            print(f"  -> {result}")

    print(f"\n\n{'='*78}\n=== Summary (wall_time_no_obs = wall_time - success_obs_time) ===\n{'='*78}")
    header = (f"{'config':<28} {'success':<9} {'avg_cnt':<10} {'avg_chunks':<12} "
              f"{'wall(s)':<10} {'obs(s)':<10} {'wall_no_obs(s)':<14} {'topp_fail':<10}")
    print(header)
    print("-" * len(header))
    for cfg_name, _, _ in configurations:
        runs = all_results[cfg_name]
        ok = [r for r in runs if "error" not in r]
        if not ok:
            print(f"{cfg_name:<28} (all errored)")
            continue
        n = len(ok)
        succ_rate = sum(r["success"] for r in ok) / n
        avg_cnt = sum(r["take_action_cnt"] for r in ok) / n
        avg_chunks = sum(r["n_chunk_calls"] for r in ok) / n
        avg_wall = sum(r["wall_time"] for r in ok) / n
        avg_obs = sum(r.get("success_obs_time", 0.0) for r in ok) / n
        avg_wall_no_obs = sum(r.get("wall_time_no_obs", r["wall_time"]) for r in ok) / n
        avg_fail = sum(r["topp_fail"] for r in ok) / n
        print(
            f"{cfg_name:<28} {succ_rate*100:>6.1f}%  {avg_cnt:<10.1f} "
            f"{avg_chunks:<12.1f} {avg_wall:<10.2f} {avg_obs:<10.2f} "
            f"{avg_wall_no_obs:<14.2f} {avg_fail:<10.2f}"
        )


if __name__ == "__main__":
    main()
