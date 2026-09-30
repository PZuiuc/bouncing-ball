#!/usr/bin/env python3
"""
Generate synthetic ball-on-conveyor trajectories that look like the real
recorded clips (dataset_conveyor_v2), but with a SHORT side wall (bounded
height -- a real bounce can clear over the top, unlike the real rig's
effectively-infinite wall) and with shape/friction/restitution sampled
around the values we already fit from real data per ball type, so the
synthetic distribution sits in the same physical regime instead of an
arbitrary one.

Uses analytic_rollout.py's OWN batched contact solver (the same physics
as the v6 pipeline) as the generator, so:
  - v6 can be evaluated on this data as a consistency check (it should
    recover the injected ground-truth params via the same CEM fitting
    pipeline used on real data).
  - contact_switch can be retrained/tested on data that (a) is in the
    right physical regime, and (b) actually contains wall contacts, to
    see whether its poor real-data transfer was regime mismatch (fixable
    with better data) or an architectural gap (floor-only, no wall term
    at all -- unfixable by data alone; see train_contact_conveyor.py).

Each ball TYPE shares ONE FIXED physics parameter set -- exactly the
real fitted ball_params_{red,yellow,egg}.json values, no per-scene
jitter -- mirroring the real setup (same physical ball, many different
tosses) so a "large adapt" (global multi-clip CEM fit, like
fit_global_ball_params.py) across all scenes of a type has a single,
well-defined ground truth to recover. Only initial conditions (position,
velocity, omega0) and per-scene wall_height/belt_speed vary.

Output: one npz per scene in dataset_conveyor_v2's own schema (t, pos,
vel, radius, object_type_code, clip, pass_id, split, record_hz) so
contact_switch's load_scenes() works UNCHANGED, plus extra ground-truth
fields (semi_axes, mu_floor, mu_wall, e_floor, e_wall, wall_height_m,
belt_speed_x, omega0) AND the full per-frame ground-truth quat/omega/vel
trajectories (quat_gt, omega_gt, vel_gt) so a demo can plug in the true
orientation/spin directly, with no adaptation, to isolate physics-only
prediction quality from estimation error.
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from analytic_rollout import rollout, Plane

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "synth_conveyor_v1"
N_SCENES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

MASS = 0.05
G = 9.81
WALL_NEG_Y, WALL_POS_Y = -0.138, 0.140
DT = 1.0 / 30.0
BALL_NAMES = ["red", "yellow", "egg"]

# real-fit regimes (mean, jitter as relative std) per ball type, loaded
# from the same ball_params/*.json used by the real-data pipeline
REAL_PARAMS = {}
for name in BALL_NAMES:
    with open(f"ball_params/ball_params_{name}.json") as f:
        p = json.load(f)
    REAL_PARAMS[name] = p


def sample_scene_params():
    name = rng.choice(BALL_NAMES)
    p = REAL_PARAMS[name]
    # FIXED per ball type -- exactly the real fitted values, no jitter --
    # so a global multi-clip fit across scenes of this type has one true
    # answer to recover.
    semi_axes = np.array(p["semi_axes_m"], dtype=np.float64)
    mu_floor = float(p["mu_floor"])
    mu_wall = float(p["mu_wall"])
    e_floor = float(p["e_floor"])
    e_wall = float(p["e_wall"])
    radius = float(semi_axes.mean())
    # SHORT wall: top edge somewhere between ~1x and ~3.5x ball radius
    # above the floor -- tall enough to matter for rolling/low bounces,
    # short enough that a real bounce apex often clears it (unlike the
    # real rig's effectively-infinite wall).
    wall_height = float(rng.uniform(1.0, 3.5) * radius)
    belt_speed_x = float(rng.uniform(0.0, 0.22)) if rng.random() < 0.7 else 0.0
    return dict(ball_name=name, semi_axes=semi_axes, mu_floor=mu_floor, mu_wall=mu_wall,
                e_floor=e_floor, e_wall=e_wall, radius=radius, wall_height=wall_height,
                belt_speed_x=belt_speed_x)


def sample_initial_state(radius):
    # start airborne over the belt, moving toward a random y-direction so
    # a meaningful fraction of scenes actually reach a wall
    x0 = rng.uniform(-0.15, 0.35)
    y0 = rng.uniform(-0.06, 0.06)
    z0 = rng.uniform(0.12, 0.30)
    vx0 = rng.uniform(-0.3, 0.3)
    vy_dir = rng.choice([-1.0, 1.0])
    vy0 = vy_dir * rng.uniform(0.05, 0.35)
    vz0 = rng.uniform(-0.3, 0.2)
    # modest initial spin -- real-regime, not the degenerate huge values
    # a short-window CEM can converge to on free-fall-only data
    omega0 = rng.uniform(-8, 8, size=3)
    return np.array([x0, y0, z0]), np.array([vx0, vy0, vz0]), omega0


def make_planes(belt_speed_x, wall_height):
    floor = Plane(torch.tensor([0., 0., 1.]), torch.tensor(0.0),
                 torch.tensor([[1., 0., 0.], [0., 1., 0.]]),
                 velocity=torch.tensor([belt_speed_x, 0., 0.]))
    wall_a = Plane(torch.tensor([0., 1., 0.]), torch.tensor(WALL_NEG_Y),
                  torch.tensor([[1., 0., 0.], [0., 0., 1.]]), height=wall_height)
    wall_b = Plane(torch.tensor([0., -1., 0.]), torch.tensor(-WALL_POS_Y),
                  torch.tensor([[1., 0., 0.], [0., 0., 1.]]), height=wall_height)
    return [floor, wall_a, wall_b]


T_FRAMES = 150
n_wall_contacts_total = 0
manifest = []
for i in range(N_SCENES):
    sp = sample_scene_params()
    pos0, v0, omega0 = sample_initial_state(sp["radius"])
    planes = make_planes(sp["belt_speed_x"], sp["wall_height"])

    pos0_t = torch.from_numpy(pos0).unsqueeze(0)
    quat0_t = torch.tensor([[1., 0., 0., 0.]])
    v0_t = torch.from_numpy(v0).unsqueeze(0)
    omega0_t = torch.from_numpy(omega0).unsqueeze(0)
    semi_axes_t = torch.from_numpy(sp["semi_axes"]).unsqueeze(0)
    mu_list = [torch.tensor([sp["mu_floor"]]), torch.tensor([sp["mu_wall"]]), torch.tensor([sp["mu_wall"]])]
    e_list = [torch.tensor([sp["e_floor"]]), torch.tensor([sp["e_wall"]]), torch.tensor([sp["e_wall"]])]
    mass_t = torch.tensor([MASS])

    traj, quat_traj, omega_traj, vel_traj = rollout(pos0_t, quat0_t, v0_t, omega0_t, semi_axes_t, mu_list, e_list,
                                                     mass_t, planes, G, DT, T_FRAMES, return_full=True)
    traj = traj[0].numpy(); quat_traj = quat_traj[0].numpy()
    omega_traj = omega_traj[0].numpy(); vel_traj = vel_traj[0].numpy()

    # stop the scene if the ball has left the belt footprint (fell off
    # the near/far edge or flew away) -- keeps clips comparable in scale
    # to the real ~1-2s recorded segments
    off_belt = (traj[:, 0] < -0.25) | (traj[:, 0] > 0.6) | (np.abs(traj[:, 1]) > 0.30) | (traj[:, 2] < -0.05)
    end = int(np.argmax(off_belt)) if off_belt.any() else T_FRAMES
    end = max(end, 20)
    pos = traj[:end]
    quat_gt = quat_traj[:end]
    omega_gt = omega_traj[:end]
    vel_gt = vel_traj[:end]
    vel = np.gradient(pos, DT, axis=0)  # finite-difference "observed" velocity, matches real-data convention

    contact_pt_y_neg = pos[:, 1] - sp["radius"]
    contact_pt_y_pos = pos[:, 1] + sp["radius"]
    hit_wall = ((contact_pt_y_neg <= WALL_NEG_Y + 1e-3) | (contact_pt_y_pos >= WALL_POS_Y - 1e-3)) & \
              (pos[:, 2] <= sp["wall_height"])
    n_wall_contacts_total += int(hit_wall.sum() > 0)

    split = "train" if i < int(0.7 * N_SCENES) else ("validation" if i < int(0.85 * N_SCENES) else "test")
    clip = f"synth-{SEED:03d}-{i:04d}"
    fname = os.path.join(OUT_DIR, f"seg{i:04d}_{clip}_pass0.npz")
    np.savez(fname,
             t=np.arange(end) * DT, pos=pos.astype(np.float32), vel=vel.astype(np.float32),
             radius=sp["radius"], object_type_code=BALL_NAMES.index(sp["ball_name"]),
             clip=clip, pass_id=0, split=split, record_hz=30.0,
             semi_axes_m=sp["semi_axes"], mu_floor=sp["mu_floor"], mu_wall=sp["mu_wall"],
             e_floor=sp["e_floor"], e_wall=sp["e_wall"], wall_height_m=sp["wall_height"],
             belt_speed_x=sp["belt_speed_x"], omega0=omega0,
             quat_gt=quat_gt.astype(np.float32), omega_gt=omega_gt.astype(np.float32),
             vel_gt=vel_gt.astype(np.float32))
    manifest.append(dict(idx=i, ball_name=sp["ball_name"], T=end, split=split,
                         wall_height_m=round(sp["wall_height"], 4), hit_wall=bool(hit_wall.sum() > 0)))

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"generated {N_SCENES} scenes in {OUT_DIR}/")
print(f"scenes with a genuine wall contact: {n_wall_contacts_total}/{N_SCENES}")
by_split = {}
for m in manifest:
    by_split[m["split"]] = by_split.get(m["split"], 0) + 1
print(f"split counts: {by_split}")
