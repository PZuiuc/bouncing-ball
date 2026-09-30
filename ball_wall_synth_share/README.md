# Synthetic ball-with-side-walls trajectories + generation code

From the `ball_pointcloud_dynamics` (PhI_iMODE) project. A ball bounces on
a floor plane between two fixed-height side walls (`WALL_NEG_Y=-0.138m`,
`WALL_POS_Y=0.140m`), physically simulated with the project's own analytic
contact solver (`analytic_rollout.py`) — no neural network involved in
generating this data, it's ground-truth physics.

Note: the generator has a `belt_speed_x` field (small horizontal surface
velocity, sampled per scene, not exactly zero) — check the metadata CSVs
if that matters for your use.

## trajectories_csv/

Five example scenes (`seg0000`-`seg0004`), one CSV pair each:

- `*_trajectory.csv`: one row per recorded frame (`record_hz`, see
  metadata) — `t` (time), `pos_x/y/z` (center of mass), `vel_x/y/z`,
  `quat_x/y/z/w` (orientation), `omega_x/y/z` (angular velocity).
- `*_metadata.csv`: fixed per-scene physical parameters — ball radius,
  object type, floor/wall friction (`mu_floor`/`mu_wall`) and restitution
  (`e_floor`/`e_wall`), wall height, belt speed, initial angular velocity,
  ellipsoid semi-axes, recording rate.

## generation_code/

- `generate_synthetic_conveyor_data.py`: the scene generator — samples
  initial conditions and per-scene wall/belt parameters, rolls out
  physics, writes one npz per scene.
- `analytic_rollout.py`: the physics engine itself (contact solver,
  `Plane` primitive, `rollout()` function) that both this generator and
  the project's real-data fitting pipeline share.
- `ball_params/ball_params_{red,yellow,egg}.json`: real-fitted physical
  parameters (mass, friction, restitution, shape) per ball type, sampled
  around by the generator so synthetic scenes sit in the same physical
  regime as the real recorded data.
