"""Offline integration checks against the real upstream optimizers and URDF."""
import json,sys,unittest
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dex_hand_retargeting import DexHandRetargeter,canonical_landmarks,operator_frame,LANDMARK_NAMES,FLEXION_JOINTS,FINGER_LINKS

ROOT=Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT/'generated/hand.xml').exists():
            from prepare_model import prepare
            prepare()
        cls.model=mujoco.MjModel.from_xml_path(str(ROOT/'generated/hand.xml'))

    def test_named_25_to_21_and_upstream_frame(self):
        # Deterministic landmarks; added metacarpals must not shift indices.
        points=np.random.default_rng(7).normal(size=(21,3))*.04
        nodes=dict(zip(LANDMARK_NAMES,points))
        for finger in ('index','middle','ring','little'):nodes[finger+'_metacarpal']=[9,9,9]
        converted=canonical_landmarks(nodes)
        moved=points@Rotation.from_euler('xyz',[.4,-.2,.7]).as_matrix().T+[2,-1,4]
        np.testing.assert_allclose(canonical_landmarks(dict(zip(LANDMARK_NAMES,moved))),converted,atol=1e-12)
        np.testing.assert_allclose(converted[0],0,atol=1e-12)
        np.testing.assert_allclose(operator_frame(points).T@operator_frame(points),np.eye(3),atol=1e-12)
        with self.assertRaises(ValueError):canonical_landmarks({'wrist':[0,0,0]})
        with self.assertRaises(ValueError):operator_frame(np.zeros((21,3)))

    def test_official_optimizers_and_calibration_gate(self):
        for method in ('vector','dexpilot'):
            with self.subTest(method=method):
                adapter=DexHandRetargeter(self.model,method=method)
                seq=adapter.retargeting;opt=seq.optimizer;robot=opt.robot
                self.assertEqual(type(opt).__name__,'VectorOptimizer' if method=='vector' else 'DexPilotOptimizer')
                self.assertEqual(len(opt.target_joint_names),20)
                self.assertEqual(len(opt.idx_pin2fixed),2)
                with self.assertRaises(ValueError):adapter.retarget({'source':'qnbot_raw_calibration_preview','skeleton_ready':False})
                # Known reachable robot pose -> real reference vectors -> solver.
                desired=np.zeros(robot.dof)
                for i,name in enumerate(robot.dof_joint_names):
                    if name in opt.target_joint_names:desired[i]=.3
                desired=np.clip(desired,robot.joint_limits[:,0],robot.joint_limits[:,1])
                robot.compute_forward_kinematics(desired)
                def position(name):return robot.get_link_pose(robot.get_link_index(name))[:3,3].copy()
                origins=np.array([position(n) for n in opt.origin_link_names])
                tasks=np.array([position(n) for n in opt.task_link_names])
                reference=tasks-origins
                for _ in range(40):q=seq.retarget(reference,fixed_qpos=adapter.fixed)
                self.assertTrue(np.isfinite(q).all())
                self.assertTrue((q>=robot.joint_limits[:,0]-1e-6).all())
                self.assertTrue((q<=robot.joint_limits[:,1]+1e-6).all())
                np.testing.assert_allclose(q[opt.idx_pin2fixed],0,atol=1e-9)
                robot.compute_forward_kinematics(q)
                actual=np.array([position(n) for n in opt.task_link_names])-np.array([position(n) for n in opt.origin_link_names])
                error=float(np.linalg.norm(actual-reference,axis=1).mean())
                print(json.dumps({'optimizer':type(opt).__name__,'mean_reference_error_m':error}))
                if method=='vector':self.assertLess(error,.02)
                # Joint-name mapping matches actual MuJoCo addresses.
                for pin_index,name in enumerate(seq.joint_names):
                    self.assertEqual(adapter.sim_indices[pin_index],self.model.joint(name).qposadr[0])

    def test_no_hyperextension_for_both_optimizers(self):
        for method in ('vector','dexpilot'):
            with self.subTest(method=method):
                adapter=DexHandRetargeter(self.model,method=method)
                seq=adapter.retargeting;opt=seq.optimizer;robot=opt.robot
                flex=np.isin(robot.dof_joint_names,FLEXION_JOINTS)
                np.testing.assert_array_equal(robot.joint_limits[flex,0],0.)
                target_flex=np.isin(opt.target_joint_names,FLEXION_JOINTS)
                np.testing.assert_array_equal(opt.opt.get_lower_bounds()[target_flex],0.)
                # A target generated from backwards-bent joints is deliberately
                # outside the allowed range; the optimizer must respect bounds.
                forbidden=np.zeros(robot.dof);forbidden[flex]=-.3
                robot.compute_forward_kinematics(forbidden)
                def positions(names):
                    return np.array([robot.get_link_pose(robot.get_link_index(n))[:3,3].copy() for n in names])
                reference=positions(opt.task_link_names)-positions(opt.origin_link_names)
                for _ in range(20):
                    q=seq.retarget(reference,fixed_qpos=adapter.fixed)
                    self.assertTrue(np.isfinite(q).all())
                    self.assertGreaterEqual(q[flex].min(),0.)
                # Abduction/opposition are still free to move in both directions.
                for name in ('link_004_joint','link_008_joint','link_012_joint','link_016_joint','link_020_joint'):
                    i=robot.dof_joint_names.index(name)
                    self.assertLess(robot.joint_limits[i,0],0.)
                    self.assertGreater(robot.joint_limits[i,1],0.)

    def test_full_calibrated_packet_and_duplicate_sequence(self):
        adapter=DexHandRetargeter(self.model)
        robot=adapter.retargeting.optimizer.robot
        robot.compute_forward_kinematics(np.zeros(robot.dof))
        links=['link_003']+[name for chain in FINGER_LINKS for name in chain]
        nodes={name:robot.get_link_pose(robot.get_link_index(link))[:3,3].tolist() for name,link in zip(LANDMARK_NAMES,links)}
        for seq in range(1,6):
            packet={'skeleton_ready':True,'skeleton_sequence':seq,'skeleton_nodes':nodes}
            result=adapter.retarget(packet)
        self.assertEqual(adapter.updates,5)
        self.assertTrue(np.isfinite(result).all())
        result[:]=99
        repeated=adapter.retarget(packet)
        self.assertEqual(adapter.updates,5)
        self.assertLess(np.max(np.abs(repeated)),4)
        self.assertEqual(repeated[self.model.joint('link_002_joint').qposadr[0]],0)
        self.assertEqual(repeated[self.model.joint('link_003_joint').qposadr[0]],0)

    def test_direct_skeleton_entry_matches_packet_entry(self):
        direct = DexHandRetargeter.from_project()
        live = DexHandRetargeter(self.model)
        robot = live.retargeting.optimizer.robot
        links = ['link_003'] + [name for chain in FINGER_LINKS for name in chain]

        for sequence, flexion in enumerate((0.1, 0.2, 0.3)):
            angles = np.array([flexion if name in FLEXION_JOINTS else 0.
                               for name in live.retargeting.joint_names])
            robot.compute_forward_kinematics(angles)
            nodes = {name: robot.get_link_pose(robot.get_link_index(link))[:3, 3].copy()
                     for name, link in zip(LANDMARK_NAMES, links)}
            expected = live.retarget(dict(skeleton_ready=True, skeleton_nodes=nodes,
                                          skeleton_sequence=sequence))
            np.testing.assert_allclose(direct.retarget_skeleton(nodes), expected, atol=1e-9)

        self.assertEqual(direct.updates, 3)
        for index, name in enumerate(direct.joint_names):
            self.assertEqual(index, self.model.joint(name).qposadr[0])


if __name__=='__main__':unittest.main()
