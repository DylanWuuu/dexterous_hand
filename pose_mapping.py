"""Fit a session-specific correction to Dex targets from measured pose anchors."""
import json
from hashlib import sha256

import numpy as np
from scipy.optimize import minimize


POSES = (
    ('open', '五指自然张开'),
    ('mcp90', '四指指根弯曲约 90°，中节自然伸展'),
    ('mcp90_pip90', '四指指根和中节同时弯曲'),
    ('pinch_index', '拇指与食指对指'),
    ('pinch_middle', '拇指与中指对指'),
    ('pinch_ring', '拇指与无名指对指'),
    ('pinch_pinky', '拇指与小指对指'),
)
SAMPLE_COUNT = 12


def pose_feature(points):
    segments = np.diff(points[1:].reshape(5, 4, 3), axis=1)
    lengths = np.linalg.norm(segments, axis=2, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError('骨骼含退化的指节，不能拟合映射')
    return (segments / lengths).ravel()


class PoseMapping:
    def __init__(self, anchors, residual_weights, bandwidth):
        self.anchors = np.asarray(anchors, dtype=float)
        self.residual_weights = np.asarray(residual_weights, dtype=float)
        self.bandwidth = float(bandwidth)
        if (self.anchors.shape != (len(POSES), 45)
                or self.residual_weights.shape != (len(POSES), 21, 3)
                or not np.isfinite(self.anchors).all()
                or not np.isfinite(self.residual_weights).all()
                or not np.isfinite(self.bandwidth) or self.bandwidth <= 0):
            raise ValueError('Dex 手势映射参数无效')

    @classmethod
    def fit(cls, measured, reference):
        anchors = np.array([pose_feature(points) for points in measured])
        distance = np.linalg.norm(anchors[:, None] - anchors[None, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        nearest = distance.min(axis=1)
        if nearest.min() < .08:
            raise ValueError('两组参考手势的骨骼过于相似，请重采并区分动作')
        bandwidth = float(np.median(nearest))
        np.fill_diagonal(distance, 0.)
        kernel = np.exp(-.5 * (distance / bandwidth) ** 2)
        residual = np.asarray(reference) - np.asarray(measured)
        weights = np.linalg.solve(kernel + .03 * np.eye(len(anchors)), residual.reshape(len(anchors), -1))
        return cls(anchors, weights.reshape(len(anchors), 21, 3), bandwidth)

    def apply(self, points):
        distance = np.linalg.norm(self.anchors - pose_feature(points), axis=1)
        kernel = np.exp(-.5 * (distance / self.bandwidth) ** 2)
        return points + np.einsum('i,ijk->jk', kernel, self.residual_weights)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix('.tmp')
        staged.write_text(json.dumps(dict(anchors=self.anchors.tolist(),
                                           residual_weights=self.residual_weights.tolist(),
                                           bandwidth=self.bandwidth)), encoding='utf-8')
        staged.replace(path)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(path.read_text(encoding='utf-8')))


def mapping_path(home, model_path, method, *, collision):
    """Use a mapping only with the exact saved calibration and robot model."""
    records = sorted((home / 'glove' / 'calibrations').glob('*/record.yaml'))
    digest = sha256(model_path.read_bytes())
    digest.update(b'pose-mapping-v1')
    digest.update(bytes([collision]))
    for record in records:
        digest.update(record.read_bytes())
        artifact = record.parent / 'artifacts' / 'hand_system.yaml'
        if artifact.is_file():
            digest.update(artifact.read_bytes())
    return home / 'dex_pose_mappings' / f'{method}-{digest.hexdigest()}.json'


def robot_references(robot, chains, limits, collision):
    """Build robot-side anchor targets from FK and reachable tip proximity."""
    names = robot.dof_joint_names
    link_ids = [robot.get_link_index('link_003')]
    link_ids += [robot.get_link_index(name) for chain in chains for name in chain]

    def points(q):
        robot.compute_forward_kinematics(q)
        value = np.array([robot.get_link_pose(i)[:3, 3].copy() for i in link_ids])
        return value - value[0]

    def feasible(q):
        return collision is None or collision.distances(q).min() >= -collision.tolerance

    neutral = np.zeros(len(names))
    base_bend = neutral.copy()
    for index in (9, 13, 17, 21):
        base_bend[names.index(f'link_{index:03d}_joint')] = 1.40
    middle_bend = base_bend.copy()
    for index in (10, 14, 18, 22):
        middle_bend[names.index(f'link_{index:03d}_joint')] = 1.20
    poses = [neutral]
    for q in (base_bend, middle_bend):
        q = np.clip(q, limits[:, 0], limits[:, 1])
        for fraction in np.linspace(1., .25, 16):
            candidate = q * fraction
            if feasible(candidate):
                poses.append(candidate)
                break
        else:
            raise ValueError('机器人弯曲参考姿态不可达')

    for finger in range(1, 5):
        start = 8 + 4 * (finger - 1)
        active = [names.index(f'link_{i:03d}_joint') for i in (*range(4, 8), *range(start, start + 4))]

        def full(x):
            q = neutral.copy()
            q[active] = x
            return q

        def loss(x):
            landmarks = points(full(x))
            difference = landmarks[4] - landmarks[4 + 4 * finger]
            return np.dot(difference, difference) / .01**2 + .01 * np.dot(x, x)

        constraints = []
        if collision is not None:
            def distance_gradient(x):
                return collision.distances(full(x), derivatives=True)[1][:, active] / .01

            constraints.append(dict(type='ineq',
                                    fun=lambda x: (collision.distances(full(x)) - collision.clearance) / .01,
                                    jac=distance_gradient))
        result = minimize(loss, neutral[active], method='SLSQP', bounds=limits[active],
                          constraints=constraints, options={'maxiter': 65, 'ftol': 1e-7})
        q = full(result.x)
        if not np.isfinite(q).all() or not feasible(q) or loss(result.x) >= loss(neutral[active]):
            raise ValueError(f'未找到可用的机器人捏合参考：{POSES[finger + 2][1]}')
        poses.append(q)

    return np.array([points(q) for q in poses])
