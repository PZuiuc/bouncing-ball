# Synthetic PyBullet sphere-bounce trajectories + generation code

From the `ball_pointcloud_dynamics` (PhI_iMODE) project's original
synthetic dataset — the "Bounce and Learn" (Purushwalkam et al., ICLR
2019) recipe: a single sphere bounces on a static floor plane, simulated
in PyBullet, with randomly sampled initial position/velocity and randomly
sampled physical parameters (friction, restitution) per scene.

Unlike the paper's original recipe (which zeroed angular velocity and
friction), this version includes real spin and friction, since the
project also needs rotational dynamics, not just translational bounce
physics.

## trajectories_csv/

Five example scenes (`scene_00000`-`scene_00004`), one CSV pair each:

- `*_trajectory.csv`: one row per recorded frame — `center_x/y/z` (ball
  position), `lin_vel_x/y/z`, `quat_x/y/z/w` (orientation), `ang_vel_x/y/z`
  (spin), `marker_dir_x/y/z` (world-frame direction of a body-fixed
  marker point — a proxy for a visible surface marking like a
  basketball's seam, since a bare sphere has no observable orientation on
  its own), `theta`/`phi` (that marker direction's spherical-coordinate
  angles — the 2 degrees of orientation actually observable from a single
  marker; the 3rd rotational DOF, spin about the marker's own axis, isn't
  observable this way).
- `*_metadata.csv`: fixed per-scene physical parameters — ball radius,
  mass, ball/floor friction and restitution coefficients, recording rate.

## generation_code/

- `generate_ball_bounce.py`: the scene generator — samples physics
  parameters and initial conditions, runs the PyBullet simulation,
  writes `trajectory.npz` (state), `pointclouds.npz` (full-sphere-surface
  point cloud per frame, from privileged 3D simulation state, not a
  single camera view), and `rgb.npz` (rendered frames) per scene.
- `generate_ball_bounce_setups.py`: grouped-generation variant — holds one
  randomly-sampled physics tuple fixed across many scenes per "setup"
  (for later identifiability/PCA analysis at training scale), otherwise
  identical simulation code.

Requires `pybullet` (`pip install pybullet`) to run.
