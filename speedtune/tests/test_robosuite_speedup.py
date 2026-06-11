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
# Put repo root on the path so speedup.py's lazy `from speedtune...` import resolves
# (compress_joint_path reuses speedtune.execution.chunk_ops.reconstruct_chunk).
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
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


def test_compress_joint_path_identity_and_shapes():
    """v=1 is identity; v>1 shrinks frame count and keeps the first waypoint
    (compression acts on ABSOLUTE joint angles — RoboTwin semantics)."""
    N, dof = 9, 7
    q = np.cumsum(np.ones((N, dof)) * 0.1, axis=0)  # monotone absolute joint path
    grip = np.linspace(0.0, 1.0, N)

    q1, g1 = speedup.compress_joint_path(q, grip, 1.0)
    assert q1.shape == (N, dof) and g1.shape == (N, 1)
    assert np.allclose(q1, q) and np.allclose(g1[:, 0], grip)

    q2, g2 = speedup.compress_joint_path(q, grip, 2.0)
    M = int(np.floor((N - 1) / 2.0)) + 1
    assert q2.shape == (M, dof) and g2.shape == (M, 1), (q2.shape, g2.shape)
    # first waypoint preserved (position 0); compression keeps absolute path start
    assert np.allclose(q2[0], q[0]) and np.isclose(g2[0, 0], grip[0])
    print(f"  compress_joint_path: v=1 identity, v=2 -> {M} frames, start preserved OK")


def test_compress_joint_path_preserves_geometry_not_motion_drop():
    """Sanity that compressing ABSOLUTE joint angles keeps endpoints in range
    (unlike subsampling deltas, which would drop displacement)."""
    N, dof = 11, 7
    q = np.linspace(0.0, 1.0, N).reshape(N, 1) * np.ones((1, dof))  # straight line 0->1
    grip = np.zeros(N)
    q2, _ = speedup.compress_joint_path(q, grip, 2.0)
    # every compressed waypoint stays within [q_start, q_end] — no undershoot blow-up
    assert q2.min() >= q[0].min() - 1e-9 and q2.max() <= q[-1].max() + 1e-9
    # monotone preserved
    assert np.all(np.diff(q2[:, 0]) >= -1e-9)
    print("  compress on absolute joints stays in-range + monotone OK")


if __name__ == "__main__":
    tests = [
        test_so3_exp_log_roundtrip,
        test_so3_exp_zero,
        test_integrate_zero_delta_is_identity,
        test_integrate_constant_pos_delta_is_linear,
        test_integrate_rotation_accumulates,
        test_compress_joint_path_identity_and_shapes,
        test_compress_joint_path_preserves_geometry_not_motion_drop,
    ]
    print(f"running {len(tests)} robosuite-speedup pure-function smoke tests...")
    for t in tests:
        t()
    print("ALL PASS")
    sys.exit(0)
