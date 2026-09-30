"""
Single-ball bounce generator, following the synthetic-data recipe described
in Bounce and Learn (Purushwalkam et al., ICLR 2019), Appendix B:
  "We simulate a set of sphere-to-plane collisions with the PyBullet Physics
   Engine... We initialize sphere locations and linear velocities randomly...
   Collision surfaces are oriented randomly and COR values are sampled
   uniformly... we create point clouds per simulation by picking a viewpoint
   and sampling only visible points on the sphere at each time step."

Extended beyond their original recipe (which set angular velocity and
friction to zero, since they only needed translational bounce dynamics) to
include real angular velocity and friction, since our goal is to also learn
rotational dynamics (torque), not just translational (force).

One scene = one sphere, one static floor plane, nothing else. Physically
simulated for `duration` seconds at `sim_hz`, recorded at `record_hz`.

Outputs, per scene, into <out_dir>/<scene_id>/:
  - trajectory.npz: ground-truth state per recorded frame
      center            [T,3]   center-of-mass position (world frame)
      lin_vel           [T,3]   linear velocity (world frame)
      quaternion        [T,4]   orientation, (x,y,z,w) -- PyBullet convention
      ang_vel           [T,3]   angular velocity (world frame)
      marker_dir_world  [T,3]   world-frame direction of the body-fixed
                                 marker (unit vector) -- this is the
                                 observable axis a point cloud with a visible
                                 surface marking (e.g. a basketball seam)
                                 could recover; (theta,phi) below are its
                                 spherical-coordinate angles
      theta, phi        [T]     spherical angles of marker_dir_world:
                                 theta = polar angle from +z, phi = azimuth
                                 in the xy-plane. This is the reduced 2-DOF
                                 "orientation" a symmetric ball actually
                                 allows you to observe (see note below).
      radius            scalar  sphere radius
      mass, friction, restitution  scalar physical parameters of this scene
  - pointclouds.npz: per-frame point cloud, covering the FULL sphere surface
      (both hemispheres -- we have privileged full 3D simulation state, so
      unlike a real single depth camera there's no reason to only keep the
      camera-facing half), in world coordinates, PLUS a small number of
      "marker" points sampled near marker_dir_world so orientation is
      actually recoverable from the cloud alone, mimicking a real
      basketball's seam/logo pattern. The center of mass is recoverable as
      (approximately, for the uniform-surface points) the point cloud's
      centroid; the marker patch's mean direction from that centroid gives
      the (theta, phi) orientation. Stored as an object array of [N_t, 3]
      arrays (N_t varies slightly per frame due to the random marker-patch
      sample count).
  - rgb.npz: rendered RGB frames, one per recorded timestep, same camera and
      frame count/alignment as pointclouds.npz/trajectory.npz -- for other
      projects that want image input rather than point clouds; not used by
      this project's own training (which is point-cloud/state-based)
      key: rgb [T,H,W,3] uint8
  - rgba.mp4: optional rendered video (--render_video), for quick visual
      sanity-checking only -- reuses the same rendered frames as rgb.npz

Note on orientation reduction to (theta, phi): a bare, uniformly-colored
sphere has NO observable orientation from geometry alone (this is flagged
as an open risk in our own proposal doc). We simulate a single body-fixed
marker point (like a valve stem, or the center of a logo) to make
orientation observable at all. That marker's *direction* from the sphere
center has only 2 degrees of freedom (it lives on the unit sphere S^2) --
the 3rd rotational DOF (spin around that marker's own axis) stays
unobservable with a single marker, matching the reduced (theta,phi)
representation requested for this project. If ever needed, a second,
off-axis marker would make the full 3-DOF orientation observable.
"""
import argparse
import os

import numpy as np
import pybullet as p
import pybullet_data


def quat_rotate(quat_xyzw, vec):
    """Rotate a 3-vector by a quaternion (PyBullet's (x,y,z,w) convention)."""
    return np.array(p.rotateVector(quat_xyzw, vec.tolist()))


def marker_direction(quat_xyzw, body_frame_marker=np.array([0.0, 0.0, 1.0])):
    """World-frame direction of a body-fixed marker (default: sphere's local +z pole)."""
    d = quat_rotate(quat_xyzw, body_frame_marker)
    return d / (np.linalg.norm(d) + 1e-12)


def direction_to_theta_phi(d):
    """d: unit vector [3]. theta = polar angle from +z in [0,pi], phi = azimuth in (-pi,pi]."""
    theta = np.arccos(np.clip(d[2], -1.0, 1.0))
    phi = np.arctan2(d[1], d[0])
    return theta, phi


def sample_full_sphere_points(center, radius, quat_xyzw, n_surface=300,
                               marker_body_dir=np.array([0.0, 0.0, 1.0]), n_marker=40,
                               marker_angular_radius=0.35, rng=None):
    """
    Sample points over the FULL sphere surface (both hemispheres, not
    restricted to what a single camera viewpoint could see) -- we have
    privileged full 3D simulation state, so there's no reason to throw away
    half the sphere the way a single real depth camera would have to.
    Also samples a denser patch of "marker" points near the body-fixed
    marker direction, mimicking a textured/marked ball (e.g. a basketball
    seam) that lets orientation be estimated from the cloud alone.
    """
    if rng is None:
        rng = np.random.default_rng()

    # uniform points on the unit sphere (Marsaglia method)
    raw = rng.normal(size=(n_surface, 3))
    normals = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    surface_pts = center + radius * normals

    # marker patch: points within `marker_angular_radius` (radians) of the
    # body-fixed marker direction
    marker_world_dir = marker_direction(quat_xyzw, marker_body_dir)
    raw2 = rng.normal(size=(n_marker * 10, 3))
    raw2 /= np.linalg.norm(raw2, axis=1, keepdims=True)
    ang_to_marker = np.arccos(np.clip(raw2 @ marker_world_dir, -1.0, 1.0))
    marker_normals = raw2[ang_to_marker < marker_angular_radius][:n_marker]
    marker_pts = center + radius * marker_normals

    if len(marker_pts) == 0:
        return surface_pts.astype(np.float32), 0
    all_pts = np.concatenate([surface_pts, marker_pts], axis=0)
    return all_pts.astype(np.float32), len(marker_pts)


def simulate_one_scene(rng, duration=2.0, sim_hz=240, record_hz=60,
                        arena_half_extent=3.0, fixed_physics=None):
    """fixed_physics: optional dict with any of {radius, mass, ball_friction,
    ball_restitution} to hold fixed (e.g. for a "setup" of many videos
    sharing the same physics, varying only initial conditions); any key not
    given still falls back to the usual rng.uniform sampling below."""
    client = p.connect(p.DIRECT)  # headless, no GUI needed for data generation
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / sim_hz)

    # Fixed (not randomized): PyBullet's actual contact friction/restitution
    # is some combination of BOTH bodies' values, so if the floor's values
    # also varied independently, two scenes with identical ball parameters
    # could behave differently -- confounding any later eta-vs-ball-friction
    # correlation study. Only the ball's own (mass, radius, friction,
    # restitution) vary; the floor is one fixed, known surface throughout.
    floor_friction = 0.6
    floor_restitution = 0.6
    floor_id = p.loadURDF("plane.urdf")
    p.changeDynamics(floor_id, -1, lateralFriction=floor_friction, restitution=floor_restitution)

    fixed_physics = fixed_physics or {}
    radius = float(fixed_physics.get("radius", rng.uniform(0.08, 0.15)))
    mass = float(fixed_physics.get("mass", rng.uniform(0.1, 1.0)))
    ball_friction = float(fixed_physics.get("ball_friction", rng.uniform(0.2, 0.9)))
    ball_restitution = float(fixed_physics.get("ball_restitution", rng.uniform(0.4, 0.95)))

    start_pos = [
        float(rng.uniform(-0.5, 0.5)),
        float(rng.uniform(-0.5, 0.5)),
        float(rng.uniform(0.8, 1.8)),
    ]
    lin_vel = [
        float(rng.uniform(-1.0, 1.0)),
        float(rng.uniform(-1.0, 1.0)),
        float(rng.uniform(-0.5, 0.5)),
    ]
    ang_vel = [float(rng.uniform(-6.0, 6.0)) for _ in range(3)]

    col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0.85, 0.35, 0.1, 1.0])
    ball_id = p.createMultiBody(
        baseMass=mass, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape,
        basePosition=start_pos,
    )
    p.changeDynamics(ball_id, -1, lateralFriction=ball_friction, restitution=ball_restitution,
                      rollingFriction=0.001, spinningFriction=0.001)
    p.resetBaseVelocity(ball_id, linearVelocity=lin_vel, angularVelocity=ang_vel)

    n_steps = int(duration * sim_hz)
    record_every = max(1, sim_hz // record_hz)

    cam_pos = np.array([0.0, -2.2, 1.4])

    centers, lin_vels, quats, ang_vels = [], [], [], []
    marker_dirs, thetas, phis = [], [], []
    pointclouds = []

    for step in range(n_steps):
        p.stepSimulation()
        if step % record_every != 0:
            continue
        pos, quat = p.getBasePositionAndOrientation(ball_id)
        lv, av = p.getBaseVelocity(ball_id)
        pos = np.array(pos)
        quat = np.array(quat)

        centers.append(pos)
        lin_vels.append(np.array(lv))
        quats.append(quat)
        ang_vels.append(np.array(av))

        m_dir = marker_direction(quat)
        marker_dirs.append(m_dir)
        th, ph = direction_to_theta_phi(m_dir)
        thetas.append(th)
        phis.append(ph)

        pts, _ = sample_full_sphere_points(pos, radius, quat, rng=rng)
        pointclouds.append(pts)

        # stop early if the ball has left the recording arena or gone to sleep far away
        if abs(pos[0]) > arena_half_extent or abs(pos[1]) > arena_half_extent or pos[2] < -1.0:
            break

    p.disconnect(client)

    traj = dict(
        center=np.stack(centers).astype(np.float32),
        lin_vel=np.stack(lin_vels).astype(np.float32),
        quaternion=np.stack(quats).astype(np.float32),
        ang_vel=np.stack(ang_vels).astype(np.float32),
        marker_dir_world=np.stack(marker_dirs).astype(np.float32),
        theta=np.array(thetas, dtype=np.float32),
        phi=np.array(phis, dtype=np.float32),
        radius=np.float32(radius),
        mass=np.float32(mass),
        ball_friction=np.float32(ball_friction),
        ball_restitution=np.float32(ball_restitution),
        floor_friction=np.float32(floor_friction),
        floor_restitution=np.float32(floor_restitution),
        record_hz=np.float32(record_hz),
    )
    return traj, pointclouds, cam_pos


def render_frames(traj, cam_pos, radius, width=320, height=240):
    """Re-simulate nothing; just replay the recorded (center, quaternion)
    trajectory through PyBullet's renderer to get RGB frames. Pure playback --
    touches no RNG state -- so calling this after simulate_one_scene never
    affects the reproducibility of trajectory.npz/pointclouds.npz for a given
    seed, regardless of when/whether it's called."""
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0.85, 0.35, 0.1, 1.0])
    ball_id = p.createMultiBody(baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape)

    view_matrix = p.computeViewMatrix(cameraEyePosition=cam_pos.tolist(),
                                       cameraTargetPosition=[0, 0, 0.3],
                                       cameraUpVector=[0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=width / height, nearVal=0.05, farVal=10)

    n_frames = traj["center"].shape[0]
    frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    for i in range(n_frames):
        p.resetBasePositionAndOrientation(ball_id, traj["center"][i].tolist(), traj["quaternion"][i].tolist())
        _, _, rgba, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix,
                                             renderer=p.ER_TINY_RENDERER)
        frames[i] = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
    p.disconnect(client)
    return frames


def write_video(frames, out_path, fps):
    """frames: [T,H,W,3] uint8 RGB, e.g. from render_frames -- kept separate
    from rendering so --render_video reuses the same frames already saved to
    rgb.npz instead of rendering twice."""
    import cv2

    height, width = frames.shape[1:3]
    # avc1/h264 needs a hardware encoder that isn't available on this headless box;
    # mp4v (MPEG-4 Part 2) is software-only and works everywhere, at the cost of
    # needing a re-encode for QuickTime/browser playback (see local re-encode step).
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_scenes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render_video", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.n_scenes):
        scene_dir = os.path.join(args.out_dir, f"scene_{i:05d}")
        os.makedirs(scene_dir, exist_ok=True)

        traj, pointclouds, cam_pos = simulate_one_scene(rng)

        np.savez(os.path.join(scene_dir, "trajectory.npz"), **traj)
        np.savez(os.path.join(scene_dir, "pointclouds.npz"),
                 pointclouds=np.array(pointclouds, dtype=object))

        frames = render_frames(traj, cam_pos, float(traj["radius"]))
        np.savez_compressed(os.path.join(scene_dir, "rgb.npz"), rgb=frames)

        if args.render_video:
            write_video(frames, os.path.join(scene_dir, "rgba.mp4"), float(traj["record_hz"]))

        print(f"[{i+1}/{args.n_scenes}] {scene_dir}: {traj['center'].shape[0]} frames, "
              f"radius={float(traj['radius']):.3f} mass={float(traj['mass']):.3f}")


if __name__ == "__main__":
    main()
