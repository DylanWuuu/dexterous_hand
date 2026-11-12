"""Qnbot calibrated Skeleton -> upstream dex-retargeting -> MuJoCo joints.

Upstream solvers and filtering are retained, with a local objective correction
and finger-direction targets. Flexion limits prohibit reverse bending.
"""
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from dex_retargeting.constants import OPERATOR2MANO, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from retarget_objective import FingerDirectionObjective
from collision_constraints import HandCollisionConstraints
from pose_mapping import POSES, PoseMapping, robot_references

ROOT=Path(__file__).resolve().parent
FINGERS=('thumb','index','middle','ring','pinky')
# Positive angles flex these joints; zero is the model's extension reference.
# Opposition and abduction joints retain their original signed ranges.
FLEXION_JOINTS=tuple(f'link_{i:03d}_joint' for start in (5,9,13,17,21) for i in range(start,start+3))
LANDMARK_NAMES=(
    'wrist','thumb_metacarpal','thumb_proximal','thumb_distal','thumb_tip',
    'index_proximal','index_intermediate','index_distal','index_tip',
    'middle_proximal','middle_intermediate','middle_distal','middle_tip',
    'ring_proximal','ring_intermediate','ring_distal','ring_tip',
    'little_proximal','little_intermediate','little_distal','little_tip',
)
# Fixed geometric points on the supplied CAD distal links, in metres.
# Measured from distal mesh end caps; never fitted to glove poses.
TIP_OFFSETS={
    'thumb':('link_007',[.029390,-.009059,.005600]),
    'index':('link_011',[-.005583,-.030223,.006151]),
    'middle':('link_015',[-.005900,-.030183,.004000]),
    'ring':('link_019',[-.005546,-.030250,-.004600]),
    'pinky':('link_023',[-.004757,-.030384,-.005600]),
}

# CMC/MCP -> PIP/IP -> DIP -> tip, matching human landmarks 1..20.
# Thumb metacarpal starts at the flexion axis (005), not opposition axis 004.
FINGER_LINKS=(
    ('link_005','link_006','link_007','dex_thumb_tip'),
    ('link_009','link_010','link_011','dex_index_tip'),
    ('link_013','link_014','link_015','dex_middle_tip'),
    ('link_017','link_018','link_019','dex_ring_tip'),
    ('link_021','link_022','link_023','dex_pinky_tip'),
)


def operator_frame(points):
    """Same SVD/Gram-Schmidt procedure as upstream SingleHandDetector.

    Source: example/vector_retargeting/single_hand_detector.py (MIT).
    No fit to robot poses, calibrated captures, or finger joint angles.
    """
    points=np.asarray(points,dtype=float)
    if points.shape!=(21,3) or not np.isfinite(points).all():
        raise ValueError('需要有限的 21x3 手部关键点（米）')
    palm=points[[0,5,9]]
    x_vector=palm[0]-palm[2]
    palm=palm-palm.mean(axis=0)
    _,singular,v=np.linalg.svd(palm)
    if singular[1]<1e-7:raise ValueError('掌部关键点退化，不能计算官方输入坐标系')
    normal=v[2]
    x=x_vector-np.sum(x_vector*normal)*normal
    x=x/np.linalg.norm(x)
    z=np.cross(x,normal)
    if np.sum(z*(palm[1]-palm[2]))<0:normal=-normal;z=-z
    return np.stack([x,normal,z],axis=1)

def canonical_landmarks(nodes,side='left'):
    try:points=np.asarray([nodes[name] for name in LANDMARK_NAMES],dtype=float)
    except (KeyError,TypeError) as exc:raise ValueError(f'SDK 标定骨骼缺少节点：{exc}') from exc
    points=points-points[0:1]
    return points@operator_frame(points)@OPERATOR2MANO[HandType[side]]

def prepare_retarget_urdf(model,side='left'):
    """Add semantic fixed frames in a generated copy; keep source URDF intact."""
    tree=ET.parse(ROOT/'robot_urdf/robot/robot.urdf');root=tree.getroot()
    for joint in root.findall('joint'):
        if joint.get('name') in FLEXION_JOINTS:
            limit=joint.find('limit');limit.set('lower',str(max(0.,float(limit.get('lower')))))
    for mesh in root.findall('.//mesh'):
        mesh.set('filename',str((ROOT/'robot_urdf/robot'/mesh.get('filename')).resolve()))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    wrist=data.xpos[model.body('link_003').id].copy()
    points=np.zeros((21,3))
    points[0]=wrist
    points[5]=data.xpos[model.body('link_009').id]
    points[9]=data.xpos[model.body('link_013').id]
    world_to_operator=operator_frame(points)@OPERATOR2MANO[HandType[side]]
    ET.SubElement(root,'link',name='dex_operator_root')
    joint=ET.SubElement(root,'joint',name='dex_operator_to_base',type='fixed')
    ET.SubElement(joint,'parent',link='dex_operator_root');ET.SubElement(joint,'child',link='base_link')
    ET.SubElement(joint,'origin',xyz=' '.join(map(str,-wrist@world_to_operator)),
                  rpy=' '.join(map(str,Rotation.from_matrix(world_to_operator.T).as_euler('xyz'))))
    for finger,(parent,offset) in TIP_OFFSETS.items():
        name=f'dex_{finger}_tip';ET.SubElement(root,'link',name=name)
        joint=ET.SubElement(root,'joint',name=name+'_fixed',type='fixed')
        ET.SubElement(joint,'parent',link=parent);ET.SubElement(joint,'child',link=name)
        ET.SubElement(joint,'origin',xyz=' '.join(map(str,offset)),rpy='0 0 0')
    target=ROOT/'generated/dex_hand.urdf';target.parent.mkdir(exist_ok=True)
    ET.indent(tree);tree.write(target,encoding='utf-8',xml_declaration=True)
    return target

class DexHandRetargeter:
    """Stateful retargeting for the project's left hand model."""

    @classmethod
    def from_project(cls, *, side='left', method='vector', collision=True):
        """Load the project hand model, generating it on first use."""
        model_path = ROOT / 'generated' / 'hand.xml'
        if model_path.exists():
            model = mujoco.MjModel.from_xml_path(str(model_path))
        else:
            from prepare_model import prepare

            model = prepare()
        return cls(model, side=side, method=method, collision=collision)

    def __init__(self,model,side='left',method='vector',collision=True):
        if side!='left':raise ValueError('当前自定义 URDF 只验证了左手，请使用 --side left')
        if method not in ('vector','dexpilot'):raise ValueError('重定向方法必须为 vector 或 dexpilot')
        self.side=side;self.method=method
        urdf=prepare_retarget_urdf(model,side)
        cfg=RetargetingConfig.load_from_file(ROOT/f'configs/hand_{method}.yml',override={'urdf_path':str(urdf)})
        self.retargeting=cfg.build()
        # Upstream normally expands limits by 1e-3 rad. Keep the new zero
        # lower bounds exact inside the solver, rather than clipping afterwards.
        opt=self.retargeting.optimizer
        lower=opt.opt.get_lower_bounds()
        lower[np.isin(opt.target_joint_names,FLEXION_JOINTS)]=0.
        opt.opt.set_lower_bounds(lower)
        self.limits=opt.robot.joint_limits
        self.objective=FingerDirectionObjective(opt,FINGER_LINKS)
        opt.opt.set_ftol_abs(1e-10)
        opt.opt.set_maxeval(100)
        # Explicit name mapping; neither parser's joint order is assumed.
        names=self.retargeting.joint_names
        self.sim_indices=np.array([model.joint(name).qposadr[0] for name in names],dtype=int)
        if set(names)!={model.joint(i).name for i in range(model.njnt)}:
            raise ValueError('重定向 URDF 与 MuJoCo 关节名称不一致')
        self.joint_names = tuple(name for _, name in sorted(zip(self.sim_indices, names)))
        self.fixed=np.zeros(len(self.retargeting.optimizer.idx_pin2fixed))
        self.output=np.zeros(model.nq);self.updates=0;self.last_ms=0.
        self.collision=HandCollisionConstraints(opt,FINGER_LINKS) if collision else None
        self.pose_mapping = None
        self.reset()

    def reset(self):
        """Reset the starting pose and filter before an independent sequence."""
        self.retargeting.reset()
        self.retargeting.set_qpos(np.zeros_like(self.output))
        if self.retargeting.filter is not None:self.retargeting.filter.reset()
        if self.collision is not None:self.collision.reset()
        self.sequence=None;self.output[:]=0.;self.objective.targets=None

    def fit_pose_mapping(self, captures):
        """Fit robot-space targets from this calibration's reference captures."""
        measured = np.array([
            np.median([canonical_landmarks(nodes, self.side) for nodes in captures[key]], axis=0)
            for key, _ in POSES
        ])
        robot = self.retargeting.optimizer.robot
        reference = robot_references(robot, FINGER_LINKS, self.limits, self.collision)
        self.pose_mapping = PoseMapping.fit(measured, reference)

    def retarget(self,packet):
        if not packet.get('skeleton_ready') or not packet.get('skeleton_nodes'):
            raise ValueError('等待应用现有官方标定后的实时 Skeleton；原始指尖不会输入 dex-retargeting')
        sequence = packet.get('skeleton_sequence')
        if sequence is not None and sequence == self.sequence:
            return self.output.copy()
        output = self.retarget_skeleton(packet['skeleton_nodes'])
        self.sequence = sequence
        return output

    def retarget_skeleton(self, nodes):
        """Retarget calibrated SDK nodes in metres to angles in joint_names order (rad)."""
        points=canonical_landmarks(nodes,self.side)
        if self.pose_mapping is not None:
            points = self.pose_mapping.apply(points)
        self.objective.set_targets(points)
        indices=self.retargeting.optimizer.target_link_human_indices
        ref_value=points[indices[1]]-points[indices[0]]
        started=time.perf_counter()
        qpos=self.retargeting.retarget(ref_value,fixed_qpos=self.fixed)
        if not np.isfinite(qpos).all():raise ValueError('官方重定向输出了无效关节角')
        qpos=np.clip(qpos,self.limits[:,0],self.limits[:,1])
        if self.collision is not None:
            candidate=qpos
            qpos=self.collision.accept(self.output[self.sim_indices],candidate)
            # Do not feed every low-pass output back as the solver's initial
            # state: that adds another lag on top of upstream temporal terms.
            if self.collision.last_scale<1.:
                self.retargeting.set_qpos(qpos)
                if self.retargeting.filter is not None:self.retargeting.filter.y[:]=qpos
        self.output[self.sim_indices]=qpos
        self.last_ms=(time.perf_counter()-started)*1000
        self.sequence = None
        self.updates += 1
        return self.output.copy()
