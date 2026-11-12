import unittest
import numpy as np
import test_retarget_objective as fixtures
from dex_hand_retargeting import DexHandRetargeter,FINGER_LINKS,FLEXION_JOINTS,LANDMARK_NAMES


class FollowResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ObjectiveTests.setUpClass();cls.model=fixtures.ObjectiveTests.model

    def test_small_flexion_change_remains_visible_with_collision(self):
        adapter=DexHandRetargeter(self.model);robot=adapter.retargeting.optimizer.robot
        q=np.zeros(robot.dof)
        q[np.isin(robot.dof_joint_names,FLEXION_JOINTS)]=.2
        def packet(sequence):
            robot.compute_forward_kinematics(q)
            links=['link_003']+[n for chain in FINGER_LINKS for n in chain]
            points=[robot.get_link_pose(robot.get_link_index(n))[:3,3].copy() for n in links]
            return {'skeleton_ready':True,'skeleton_sequence':sequence,'skeleton_nodes':dict(zip(LANDMARK_NAMES,points))}
        for i in range(40):before=adapter.retarget(packet(i))
        q[robot.dof_joint_names.index('link_009_joint')]+=.003
        for i in range(40,80):after=adapter.retarget(packet(i))
        self.assertGreater(np.linalg.norm(after-before),1e-4)
        self.assertGreaterEqual(adapter.collision.distances(after[adapter.sim_indices]).min(),-adapter.collision.tolerance)

    def test_constraint_retains_numerical_clearance(self):
        adapter=DexHandRetargeter(self.model);c=adapter.collision
        x=np.zeros(adapter.retargeting.optimizer.opt_dof)
        result=np.zeros(len(c.pairs));c.constraint(result,x,np.empty(0))
        d=c.distances(np.zeros(c.robot.dof))
        np.testing.assert_allclose(result,(c.clearance-d)/.01)


if __name__=='__main__':unittest.main()
