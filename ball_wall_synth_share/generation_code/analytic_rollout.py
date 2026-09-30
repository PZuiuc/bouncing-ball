#!/usr/bin/env python3
"""
Our own contact simulator -- no LCP, no lemkelcp, no ElasticLCP. Directly
implements the closed-form per-contact resolution we derived by hand:

  1. Compression impulse: lambda_n_c = -v_n / (Jn Mi Jn^T), the impulse that
     zeros the normal velocity (same 1D "try x=0, else solve Mx+q=0" case,
     since Jn/Jt are decoupled for a sphere).
  2. Restitution: lambda_n_total = (1+e) * lambda_n_c (Newton's rule).
  3. Stick/slip, checked against the FULL restituted budget (the fix from
     the ContactNets work, not the naive small-bound check):
       stick_target = -(Jt Mi Jt^T)^-1 (Jt f)   [unscaled, independent of
                                                   lambda_n for a sphere]
       bound = mu * lambda_n_total
       lambda_t = stick_target if |stick_target| <= bound else bound * dir
  4. Apply: v += Mi (Jn^T lambda_n_total + Jt^T lambda_t).

Multiple simultaneous contacts (floor + wall) are resolved by looping over
active planes and applying each contact's impulse in sequence within the
same step (Gauss-Seidel style) -- an approximation to a true joint solve,
but the standard one used by most real-time engines, and unlike
ContactNets' resolver, genuinely supports more than one plane at all.

Fully vectorized over a batch dimension (CEM candidates) -- this is our own
code, so there's no ElasticLCP-style batch_n>1 bug to work around.
"""
import torch


def quat_to_rotmat(q):
    """q: (...,4) wxyz, UNIT-normalized here (unlike ContactNets' qrot,
    which doesn't renormalize -- see QUATERNION_NOTES.md's "real footgun").
    Returns (...,3,3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def ellipsoid_contact_offset_world(R, semi_axes, normal_world):
    """The material contact-point offset from center, in world frame, for
    the given plane normal. Same derivation as EllipsoidPlane3D."""
    n_body = torch.einsum("...ji,...j->...i", R, normal_world)  # R^T @ n
    t = -n_body
    a2 = semi_axes[..., 0] ** 2
    b2 = semi_axes[..., 1] ** 2
    c2 = semi_axes[..., 2] ** 2
    num = torch.stack([a2 * t[..., 0], b2 * t[..., 1], c2 * t[..., 2]], dim=-1)
    denom = a2 * t[..., 0] ** 2 + b2 * t[..., 1] ** 2 + c2 * t[..., 2] ** 2
    k = 1.0 / denom.clamp(min=1e-12).sqrt()
    p_body = num * k.unsqueeze(-1)
    return torch.einsum("...ij,...j->...i", R, p_body)  # R @ p_body


def resolve_contact(v, omega, mass, inertia, r_offset, normal, tangents, mu, e):
    """One contact's impulse resolution, vectorized over batch. v,omega:
    (B,3). r_offset: (B,3) contact point offset from center, world frame.
    normal: (3,) or (B,3). tangents: (2,3) or (B,2,3). mu,e: (B,) or scalar.
    Returns updated (v, omega)."""
    B = v.shape[0]
    if normal.dim() == 1:
        normal = normal.unsqueeze(0).expand(B, -1)
    if tangents.dim() == 2:
        tangents = tangents.unsqueeze(0).expand(B, -1, -1)

    inv_m = 1.0 / mass
    inv_I = 1.0 / inertia

    def point_vel_jacobian_response(direction):
        """Returns (J.[v;w] scalar, J Mi J^T scalar) for a single direction,
        batched. J = [direction, r_offset x direction] (see derivation)."""
        ang_part = torch.cross(r_offset, direction, dim=-1)  # (B,3)
        j_dot_vel = (direction * v).sum(-1) + (ang_part * omega).sum(-1)
        j_mi_jt = inv_m * (direction * direction).sum(-1) + inv_I * (ang_part * ang_part).sum(-1)
        return j_dot_vel, j_mi_jt, ang_part

    # --- normal: compression + restitution ---
    vn, Mn, ang_n = point_vel_jacobian_response(normal)
    lambda_n_c = torch.clamp(-vn, min=0.0) / Mn.clamp(min=1e-9)
    lambda_n_total = (1.0 + e) * lambda_n_c

    # --- tangential: unscaled stick target (2D solve), then cap ---
    dir_t0, dir_t1 = tangents[:, 0], tangents[:, 1]
    b0, M00, ang_t0 = point_vel_jacobian_response(dir_t0)
    b1, M11, ang_t1 = point_vel_jacobian_response(dir_t1)
    # off-diagonal M01 = dir_t0 . Mi . (ang response of dir_t1) -- for a
    # sphere the two tangent directions are orthogonal and Mi is isotropic,
    # so this is diagonal; keep it simple and exact for spheres (revisit
    # for eggs, same caveat as the ContactNets fix noted).
    stick0 = -b0 / M00.clamp(min=1e-9)
    stick1 = -b1 / M11.clamp(min=1e-9)
    stick_norm = torch.sqrt(stick0 ** 2 + stick1 ** 2).clamp(min=1e-9)
    bound = mu * lambda_n_total
    scale = torch.clamp(bound / stick_norm, max=1.0)
    lam_t0 = stick0 * scale
    lam_t1 = stick1 * scale

    only_touching = (vn < 0).float()  # no impulse if already separating
    lambda_n_total = lambda_n_total * only_touching
    lam_t0 = lam_t0 * only_touching
    lam_t1 = lam_t1 * only_touching

    dv = inv_m.unsqueeze(-1) * (
        normal * lambda_n_total.unsqueeze(-1)
        + dir_t0 * lam_t0.unsqueeze(-1) + dir_t1 * lam_t1.unsqueeze(-1))
    dw = inv_I.unsqueeze(-1) * (
        ang_n * lambda_n_total.unsqueeze(-1)
        + ang_t0 * lam_t0.unsqueeze(-1) + ang_t1 * lam_t1.unsqueeze(-1))
    return v + dv, omega + dw


class Plane:
    def __init__(self, normal, offset, tangents, velocity=None, height=None):
        self.normal = normal          # (3,)
        self.offset = offset          # scalar
        self.tangents = tangents      # (2,3)
        self.velocity = velocity if velocity is not None else torch.zeros(3)  # belt motion
        # world-z above which this plane no longer blocks contact -- None
        # (default) means infinite/floor-to-ceiling, e.g. the floor and
        # the real rig's walls. A finite value models a SHORT wall: the
        # ball only collides while the contact point is below the wall's
        # top edge, and sails over it once a bounce carries it higher.
        # This is a static Python attribute (known at trace time), so
        # branching on `is not None` in _substep is safe under
        # torch.compile -- unlike branching on a tensor's runtime value.
        self.height = height

    def to(self, device):
        return Plane(self.normal.to(device), self.offset.to(device),
                    self.tangents.to(device), self.velocity.to(device), self.height)


def rollout(pos0, quat0, v0, omega0, semi_axes, mu, e, mass, planes, g, dt, n_steps, substeps=8,
           return_state=False, return_full=False):
    """Batched analytic rollout. pos0,v0,omega0: (B,3). quat0: (B,4).
    semi_axes: (B,3). mu,e: each either a single (B,) tensor (applied to
    every plane -- old behavior) or a list of (B,) tensors, one per entry
    in `planes` (different contact material per surface, e.g. belt vs.
    side wall). Returns positions (B,n_steps,3), recorded at the OUTER dt
    cadence (one row per n_steps), but physics is advanced in `substeps`
    smaller increments per outer step. If return_state=True, ALSO returns
    the final (pos,quat,v,omega) after all n_steps -- lets a caller carry
    (quat,omega) forward continuously across separate rollout calls
    (e.g. advancing spin via the model's own dynamics between reforecast
    points, instead of re-estimating it from scratch each time -- see
    contactnet_rolling_demo.py's carry-forward omega mode).

    Why: validated directly against synthetic ground truth (known true
    radius/mu/e/omega0) -- without substepping, per-bounce contact-timing
    error (checking phi<=threshold only once per fixed dt, not the exact
    impact instant) compounds across multiple bounces, especially for
    high-restitution objects (more bounces per unit time = more chances to
    compound): errors of >10 radii by frame 60 for e>0.85 scenes, despite
    tracking within 0.1-0.15r at frame 10. Finer substeps shrink each
    bounce's timing error proportionally, without needing a full
    exact-impact-time solve."""
    B = pos0.shape[0]
    inertia = 0.4 * mass * (semi_axes[:, 0]) ** 2  # sphere I=2/5 m r^2
    pos, quat, v, omega = pos0.clone(), quat0.clone(), v0.clone(), omega0.clone()
    out = torch.zeros(B, n_steps, 3, dtype=pos0.dtype, device=pos0.device)
    out_quat = torch.zeros(B, n_steps, 4, dtype=pos0.dtype, device=pos0.device) if return_full else None
    out_omega = torch.zeros(B, n_steps, 3, dtype=pos0.dtype, device=pos0.device) if return_full else None
    out_vel = torch.zeros(B, n_steps, 3, dtype=pos0.dtype, device=pos0.device) if return_full else None
    sub_dt = dt / substeps
    gravity = torch.tensor([0.0, 0.0, -g], dtype=pos0.dtype, device=pos0.device)

    for t in range(n_steps):
        out[:, t] = pos
        if return_full:
            out_quat[:, t] = quat
            out_omega[:, t] = omega
            out_vel[:, t] = v
        for _ in range(substeps):
            pos, quat, v, omega = _substep(pos, quat, v, omega, semi_axes, mu, e, mass,
                                            inertia, planes, gravity, sub_dt)
    if return_full:
        return out, out_quat, out_omega, out_vel
    if return_state:
        return out, (pos, quat, v, omega)
    return out


_GRAVITY_CACHE = {}
_STEP8_COMPILED = None


def _step8(pos, quat, v, omega, semi_axes, mu, e, mass, inertia, planes, gravity, sub_dt):
    """One OUTER dt of physics = 8 inner substeps. Fusing these into a
    single torch.compile'd call (vs. compiling _substep alone and calling
    it 8x) matters a lot: CUDA-graph replay overhead is paid once per
    call, so fusing cuts total launches 8x for identical physics."""
    for _ in range(8):
        pos, quat, v, omega = _substep(pos, quat, v, omega, semi_axes, mu, e, mass,
                                       inertia, planes, gravity, sub_dt)
    return pos, quat, v, omega


def _get_step8_compiled():
    global _STEP8_COMPILED
    if _STEP8_COMPILED is None:
        _STEP8_COMPILED = torch.compile(_step8, mode="reduce-overhead", dynamic=False)
    return _STEP8_COMPILED


def rollout_fast(pos0, quat0, v0, omega0, semi_axes, mu, e, mass, planes, g, dt, n_steps, substeps=8,
                 return_state=False):
    """GPU-compiled counterpart to rollout() -- same physics/semantics and
    same (B,n_steps,3) output, but fuses each outer step's 8 substeps into
    one torch.compile(mode='reduce-overhead') CUDA-graph call. Measured
    ~10x faster than eager rollout() on GPU for the batch sizes this
    module actually uses (CEM populations, rolling-forecast horizons) --
    see bench_compiled_rollout.py. Falls back to plain rollout() on CPU
    (cudagraph capture requires CUDA) and requires substeps==8 (the fused
    step is hardcoded for it -- this module's rollout() call sites all use
    the default anyway). return_state: see rollout()'s docstring.

    First call at a given (batch_size, n_steps) shape pays a real compile
    cost (tens of seconds) -- only worth it for shapes reused many times
    (e.g. the per-step omega0 CEM, called ~8x per forecast x ~20-30
    forecasts per segment), not one-off calls (e.g. the initial
    radius/mu/e fit, which stays on plain CPU rollout())."""
    if pos0.device.type != "cuda":
        return rollout(pos0, quat0, v0, omega0, semi_axes, mu, e, mass, planes, g, dt, n_steps, substeps,
                       return_state=return_state)
    assert substeps == 8, "rollout_fast's fused step is hardcoded for substeps=8"
    step8 = _get_step8_compiled()
    inertia = 0.4 * mass * (semi_axes[:, 0]) ** 2
    pos, quat, v, omega = pos0.clone(), quat0.clone(), v0.clone(), omega0.clone()
    sub_dt = dt / substeps
    cache_key = (g, pos0.dtype, pos0.device)
    if cache_key not in _GRAVITY_CACHE:
        _GRAVITY_CACHE[cache_key] = torch.tensor([0.0, 0.0, -g], dtype=pos0.dtype, device=pos0.device)
    gravity = _GRAVITY_CACHE[cache_key]
    out = torch.zeros(pos0.shape[0], n_steps, 3, dtype=pos0.dtype, device=pos0.device)
    for t in range(n_steps):
        out[:, t] = pos
        torch.compiler.cudagraph_mark_step_begin()
        pos, quat, v, omega = step8(pos, quat, v, omega, semi_axes, mu, e, mass, inertia, planes, gravity, sub_dt)
        pos, quat, v, omega = pos.clone(), quat.clone(), v.clone(), omega.clone()
    if return_state:
        return out, (pos, quat, v, omega)
    return out


RESTING_VN_EPS = 0.06  # m/s -- see note below; ~1.5x a single substep's
                       # unopposed gravity kick at dt=1/240s (g*dt~=0.041)


def _substep(pos, quat, v, omega, semi_axes, mu, e, mass, inertia, planes, gravity, dt):
    R = quat_to_rotmat(quat)

    # Cancel gravity's component along any contact normal where the ball
    # is ALREADY touching with ~zero incoming normal velocity -- a
    # genuine resting contact, checked BEFORE gravity is applied, using
    # this substep's starting state. Without this, gravity unconditionally
    # nudges a resting ball into the surface every substep, and
    # restitution rebounds a fraction of that fictitious "impact" back
    # out (verified directly: bounded ~1mm / ~0.1m/s chatter, never
    # growing, invisible at the 30Hz output -- but avoidable at the
    # source instead of relying on the chatter staying bounded). Walls
    # have a horizontal normal, so gravity (purely vertical) has zero
    # component along them -- this only ever actually fires for the
    # floor. Genuine bounces are untouched: a real impact has large
    # |vn|, so `resting` is False and gravity + full restitution apply
    # exactly as before.
    B = pos.shape[0]
    gravity_eff = gravity.unsqueeze(0).expand(B, -1).clone()
    for plane in planes:
        r_off_g = ellipsoid_contact_offset_world(R, semi_axes, plane.normal)
        normal_b = plane.normal.expand_as(r_off_g)
        contact_pt = pos + r_off_g
        phi = (contact_pt * normal_b).sum(-1) - plane.offset
        touching = phi <= 1e-4
        ang_part = torch.cross(r_off_g, normal_b, dim=-1)
        vn_pre = (v * normal_b).sum(-1) + (ang_part * omega).sum(-1)
        resting = touching & (vn_pre.abs() < RESTING_VN_EPS)
        g_along_n = (gravity * plane.normal).sum()
        gravity_eff = gravity_eff - resting.unsqueeze(-1).to(gravity_eff.dtype) * g_along_n * plane.normal

    v = v + gravity_eff * dt  # free-flight update (gravity gated per above)
    mu_list = mu if isinstance(mu, (list, tuple)) else [mu] * len(planes)
    e_list = e if isinstance(e, (list, tuple)) else [e] * len(planes)
    for plane, mu_i, e_i in zip(planes, mu_list, e_list):
        r_off = ellipsoid_contact_offset_world(R, semi_axes, plane.normal)
        contact_pt = pos + r_off
        phi = (contact_pt * plane.normal).sum(-1) - plane.offset
        active = phi <= 1e-4
        if plane.height is not None:
            active = active & (contact_pt[:, 2] <= plane.height)
        # NOTE: no `if not active.any(): continue` short-circuit here --
        # a data-dependent Python branch on a GPU tensor forces a
        # device->host sync AND breaks torch.compile/CUDA-graph capture
        # (control flow must be static). Always compute + mask via
        # torch.where instead; correctness is identical, just no free
        # skip when nothing touches this plane in the whole batch.
        v_rel = v - plane.velocity  # belt motion: resolve in the belt's own frame
        v_new, omega_new = resolve_contact(v_rel, omega, mass, inertia, r_off,
                                           plane.normal, plane.tangents, mu_i, e_i)
        v_new = v_new + plane.velocity
        v = torch.where(active.unsqueeze(-1), v_new, v)
        omega = torch.where(active.unsqueeze(-1), omega_new, omega)

    pos = pos + v * dt
    # quaternion integration: qdot = 0.5 * q * (0,omega)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    ox, oy, oz = omega[:, 0], omega[:, 1], omega[:, 2]
    dq = 0.5 * torch.stack([
        -x * ox - y * oy - z * oz,
        w * ox + y * oz - z * oy,
        w * oy + z * ox - x * oz,
        w * oz + x * oy - y * ox,
    ], dim=-1)
    quat = quat + dq * dt
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return pos, quat, v, omega
