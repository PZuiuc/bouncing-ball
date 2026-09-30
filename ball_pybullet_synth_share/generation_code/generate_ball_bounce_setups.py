#!/usr/bin/env python3
"""
Grouped dataset generation: n_setups groups, each with ONE randomly-sampled
(radius, mass, ball_friction, ball_restitution) tuple held FIXED across
n_videos_per_setup scenes (only initial conditions vary within a setup).
Same underlying simulate_one_scene() as generate_ball_bounce.py -- same
trajectory.npz schema, just with an added `setup_id` field so scenes can
be grouped later (for training, setup grouping doesn't matter at all --
this is for optional later identifiability/PCA analysis at training scale).
"""
import argparse
import os
import numpy as np

from generate_ball_bounce import simulate_one_scene, render_frames

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", required=True)
ap.add_argument("--n_setups", type=int, default=50)
ap.add_argument("--n_videos_per_setup", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--start_scene", type=int, default=0, help="global scene index to start numbering from (for resuming/sharding)")
ap.add_argument("--start_setup", type=int, default=0, help="setup index to start from (for resuming/sharding)")
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
setup_rng = np.random.default_rng(args.seed)
# advance setup_rng to start_setup's draw point so resumed shards get the SAME
# physics tuples a full from-scratch run would have produced for those setups
for _ in range(args.start_setup):
    setup_rng.uniform(0.08, 0.15); setup_rng.uniform(0.1, 1.0)
    setup_rng.uniform(0.2, 0.9); setup_rng.uniform(0.4, 0.95)

global_idx = args.start_scene
for setup_i in range(args.start_setup, args.n_setups):
    fixed_physics = dict(
        radius=float(setup_rng.uniform(0.08, 0.15)),
        mass=float(setup_rng.uniform(0.1, 1.0)),
        ball_friction=float(setup_rng.uniform(0.2, 0.9)),
        ball_restitution=float(setup_rng.uniform(0.4, 0.95)),
    )
    video_rng = np.random.default_rng(args.seed * 100000 + setup_i)  # deterministic per-setup IC stream
    for video_i in range(args.n_videos_per_setup):
        scene_dir = os.path.join(args.out_dir, f"scene_{global_idx:05d}")
        os.makedirs(scene_dir, exist_ok=True)

        traj, pointclouds, cam_pos = simulate_one_scene(video_rng, fixed_physics=fixed_physics)
        traj["setup_id"] = np.int32(setup_i)

        np.savez(os.path.join(scene_dir, "trajectory.npz"), **traj)
        np.savez(os.path.join(scene_dir, "pointclouds.npz"),
                 pointclouds=np.array(pointclouds, dtype=object))
        frames = render_frames(traj, cam_pos, float(traj["radius"]))
        np.savez_compressed(os.path.join(scene_dir, "rgb.npz"), rgb=frames)

        print(f"[setup {setup_i+1}/{args.n_setups} video {video_i+1}/{args.n_videos_per_setup}] "
              f"{scene_dir}: {traj['center'].shape[0]} frames, "
              f"r={fixed_physics['radius']:.3f} m={fixed_physics['mass']:.3f} "
              f"fric={fixed_physics['ball_friction']:.3f} rest={fixed_physics['ball_restitution']:.3f}",
              flush=True)
        global_idx += 1

print(f"DONE. generated {global_idx - args.start_scene} scenes across "
      f"{args.n_setups - args.start_setup} setups into {args.out_dir}")
