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
"""CLOUD-ONLY preflight for the robosuite/LIBERO joint-space speedup (方案 1).

Needs a real LIBERO + robosuite + mujoco install (i.e. run on the cloud box,
NOT locally). It does two things:

  PART A — dump the §7 facts the speedup design is gated on:
    controller type / control_delta / output_max / input_max, action_dim,
    obs keys (confirm robot0_eef_pos/quat + robot0_joint_pos exist), arm joint
    & eef-site indices.

  PART B — empirically validate rlinf/envs/libero/speedup.integrate_eef_deltas
    against the *actual* OSC: apply known delta actions, step the env, compare
    the real eef trajectory vs the integrator's prediction. A small error
    (cm / few deg, integrator slightly ahead due to OSC tracking lag) => the
    convention (axis-angle, world frame) is RIGHT. A large/wrong-direction
    error => convention mismatch (euler vs axis-angle, or body vs world);
    then adjust integrate_eef_deltas accordingly.

Run (on cloud, inside the env that has libero):
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    cd $WORKSPACE/RLinf
    python3 speedtune/tests/cloud_check_libero_integration.py --suite libero_spatial --task 0

Adjust --suite for your variant (libero_spatial/object/goal/10/90, or
liberoplus/liberopro suites). If env construction differs in your setup, only
the build_env() helper needs editing.
"""
import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rlinf.envs.libero.speedup import integrate_eef_deltas, so3_log  # noqa: E402


def quat2mat_xyzw(q):
    """robosuite quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def build_env(suite: str, task: int):
    """Construct one LIBERO env. Edit here if your variant differs."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[suite]()
    t = task_suite.get_task(task)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=128, camera_widths=128
    )
    env.seed(0)
    obs = env.reset()
    return env, obs, t.language


def _robosuite_env(env):
    """Unwrap OffScreenRenderEnv -> underlying robosuite env (best-effort)."""
    for attr in ("env", "unwrapped"):
        e = getattr(env, attr, None)
        if e is not None and hasattr(e, "robots"):
            return e
    return env if hasattr(env, "robots") else None


def part_a_dump(env, obs):
    print("\n========== PART A: §7 facts ==========")
    print("[obs keys]", sorted(obs.keys()))
    for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos",
              "robot0_gripper_qpos"):
        v = obs.get(k)
        print(f"  {k}: {None if v is None else np.asarray(v).shape}")

    rs = _robosuite_env(env)
    if rs is None:
        print("[warn] could not unwrap robosuite env; skipping controller dump")
        return rs
    robot = rs.robots[0]
    print("[action_dim]", rs.action_dim if hasattr(rs, "action_dim") else "?")
    # controller (robosuite 1.4: .controller; 1.5: .part_controllers['right'])
    ctrl = getattr(robot, "controller", None)
    if ctrl is None:
        pcs = getattr(robot, "part_controllers", None)
        ctrl = pcs.get("right") if isinstance(pcs, dict) else None
    if ctrl is not None:
        for f in ("name", "use_delta", "input_max", "input_min",
                  "output_max", "output_min", "control_dim", "ramp_ratio"):
            print(f"  controller.{f} =", getattr(ctrl, f, "<n/a>"))
    else:
        print("[warn] controller object not found; check robosuite version API")
    sim = getattr(rs, "sim", None)
    print("[sim timestep]", None if sim is None else sim.model.opt.timestep)
    return rs


def part_b_validate(env, obs, n_steps, axis):
    print("\n========== PART B: integration vs real OSC ==========")
    rs = _robosuite_env(env)
    ctrl = getattr(rs.robots[0], "controller", None) if rs else None
    if ctrl is None:
        pcs = getattr(rs.robots[0], "part_controllers", None) if rs else None
        ctrl = pcs.get("right") if isinstance(pcs, dict) else None
    out_max = np.asarray(getattr(ctrl, "output_max", np.ones(6)), dtype=np.float64)
    pos_out_max = out_max[:3]
    rot_out_max = out_max[3:6]
    print(f"[using output_max] pos={pos_out_max}, rot={rot_out_max}")

    p0 = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    R0 = quat2mat_xyzw(obs["robot0_eef_quat"])

    # small free-space delta chunk: constant unit action along one axis
    deltas = np.zeros((n_steps, 7), dtype=np.float64)
    deltas[:, axis] = 0.15  # small, stays in [-1,1]; pick a pos (0-2) or rot (3-5) axis
    deltas[:, 6] = obs["robot0_gripper_qpos"][0] if "robot0_gripper_qpos" in obs else 0.0

    # prediction (open-loop integrate, same output_max as the controller)
    pred_pos, pred_rot, _ = integrate_eef_deltas(
        p0, R0, deltas, pos_out_max, rot_out_max
    )

    # actual: step the env one delta per env.step, record eef trajectory
    act_pos = np.zeros((n_steps, 3))
    act_rot = np.zeros((n_steps, 3, 3))
    for k in range(n_steps):
        obs, _, done, _ = env.step(deltas[k])
        act_pos[k] = obs["robot0_eef_pos"]
        act_rot[k] = quat2mat_xyzw(obs["robot0_eef_quat"])
        if done:
            act_pos, act_rot = act_pos[: k + 1], act_rot[: k + 1]
            pred_pos, pred_rot = pred_pos[: k + 1], pred_rot[: k + 1]
            break

    K = act_pos.shape[0]
    pos_err = np.linalg.norm(pred_pos[:K] - act_pos, axis=1)
    rot_err = np.array([np.linalg.norm(so3_log(pred_rot[i] @ act_rot[i].T))
                        for i in range(K)])
    # direction agreement on net displacement (robust to tracking magnitude)
    net_pred = pred_pos[K - 1] - p0
    net_act = act_pos[K - 1] - p0
    cos = (net_pred @ net_act) / (np.linalg.norm(net_pred) * np.linalg.norm(net_act) + 1e-9)

    print(f"[steps compared] {K}")
    print(f"[pos err]  max={pos_err.max():.4f} m  final={pos_err[-1]:.4f} m")
    print(f"[rot err]  max={np.degrees(rot_err.max()):.2f} deg "
          f"final={np.degrees(rot_err[-1]):.2f} deg")
    print(f"[net-displacement direction cos] {cos:.3f}  (1.0 = same direction)")
    print("\nINTERPRET:")
    print("  cos≈1 and small errs  -> convention RIGHT (gap = OSC tracking lag).")
    print("  cos<0 / huge errs     -> convention WRONG: try euler instead of")
    print("                           axis-angle, or body-frame instead of world")
    print("                           (swap the few lines in integrate_eef_deltas).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--axis", type=int, default=0,
                    help="0-2 = pos x/y/z, 3-5 = rot x/y/z (axis-angle)")
    args = ap.parse_args()

    env, obs, lang = build_env(args.suite, args.task)
    print(f"[task] {args.suite}#{args.task}: {lang}")
    part_a_dump(env, obs)
    part_b_validate(env, obs, args.n_steps, args.axis)
    env.close()
    print("\nDONE.")
