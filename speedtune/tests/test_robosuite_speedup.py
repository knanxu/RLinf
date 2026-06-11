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
"""CPU smoke test for the *version-robust pure* parts of the robosuite/LIBERO
joint-space speedup (rlinf/envs/libero/speedup.py).

Covers only the numpy-only functions (SO(3) exp/log, EEF delta integration).
The mujoco IK / PD execution / orchestrator are gated on a cloud robosuite
env (no mujoco locally) and are intentionally NOT exercised here.

Run:  python3 speedtune/tests/test_robosuite_speedup.py
"""
import importlib.util
import os
import sys

import numpy as np

# Import the module by file path so we don't drag in the rlinf package __init__
# chain (which may import heavy env deps). The module's own mujoco/torch imports
# are lazy, so importing it on a CPU-only box is fine.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SPEEDUP_PATH = os.path.join(_REPO_ROOT, "rlinf", "envs", "libero", "speedup.py")
_spec = importlib.util.spec_from_file_location("_libero_speedup", _SPEEDUP_PATH)
speedup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(speedup)


def test_so3_exp_log_roundtrip():
    rng = np.random.RandomState(0)
    for _ in range(50):
        # random axis-angle with |theta| < pi to keep log unique
        w = rng.randn(3)
        w = w / (np.linalg.norm(w) + 1e-9) * rng.uniform(0.0, 3.0)
        R = speedup.so3_exp(w)
        # valid rotation: orthonormal, det 1
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
        w_rec = speedup.so3_log(R)
        assert np.allclose(w, w_rec, atol=1e-7), f"{w} vs {w_rec}"
    print("  so3_exp/so3_log roundtrip OK (50 random axis-angles)")


def test_so3_exp_zero():
    R = speedup.so3_exp(np.zeros(3))
    assert np.allclose(R, np.eye(3))
    assert np.allclose(speedup.so3_log(np.eye(3)), np.zeros(3), atol=1e-9)
    print("  so3_exp(0) == I, so3_log(I) == 0 OK")


def test_integrate_zero_delta_is_identity():
    """All-zero deltas must leave EEF pose pinned at the start pose."""
    start_pos = np.array([0.3, -0.1, 0.5])
    start_rot = speedup.so3_exp(np.array([0.1, 0.2, -0.3]))
    M = 8
    delta = np.zeros((M, 7))
    delta[:, 6] = 0.7  # gripper passthrough
    pos, rot, grip = speedup.integrate_eef_deltas(start_pos, start_rot, delta)
    assert pos.shape == (M, 3) and rot.shape == (M, 3, 3) and grip.shape == (M,)
    for k in range(M):
        assert np.allclose(pos[k], start_pos), f"pos drifted at {k}"
        assert np.allclose(rot[k], start_rot), f"rot drifted at {k}"
    assert np.allclose(grip, 0.7)
    print("  zero-delta integration pins EEF pose + passes gripper OK")


def test_integrate_constant_pos_delta_is_linear():
    """Constant positional delta → linear cumulative translation (× output_max)."""
    start_pos = np.zeros(3)
    start_rot = np.eye(3)
    M = 5
    delta = np.zeros((M, 7))
    delta[:, 0] = 1.0  # +x each frame
    pos_output_max = 0.05  # 5cm per unit action
    pos, rot, _ = speedup.integrate_eef_deltas(
        start_pos, start_rot, delta, pos_output_max=pos_output_max
    )
    for k in range(M):
        assert np.allclose(pos[k], [(k + 1) * pos_output_max, 0.0, 0.0]), pos[k]
        assert np.allclose(rot[k], np.eye(3))  # no rotation delta
    print("  constant pos-delta → linear translation (scaled by output_max) OK")


def test_integrate_rotation_accumulates():
    """Constant small rot delta about z → cumulative yaw."""
    start_rot = np.eye(3)
    M = 4
    step = 0.1  # rad per frame about +z
    delta = np.zeros((M, 7))
    delta[:, 5] = step  # axis-angle z component
    _, rot, _ = speedup.integrate_eef_deltas(np.zeros(3), start_rot, delta)
    # after M frames, total yaw ~ M*step
    total = speedup.so3_log(rot[-1])
    assert np.allclose(total, [0.0, 0.0, M * step], atol=1e-6), total
    print("  rotation-delta accumulates to cumulative yaw OK")


if __name__ == "__main__":
    tests = [
        test_so3_exp_log_roundtrip,
        test_so3_exp_zero,
        test_integrate_zero_delta_is_identity,
        test_integrate_constant_pos_delta_is_linear,
        test_integrate_rotation_accumulates,
    ]
    print(f"running {len(tests)} robosuite-speedup pure-function smoke tests...")
    for t in tests:
        t()
    print("ALL PASS")
    sys.exit(0)
