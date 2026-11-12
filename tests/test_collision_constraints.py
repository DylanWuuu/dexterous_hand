import unittest
import numpy as np
import test_retarget_objective as objective_tests
from dex_hand_retargeting import DexHandRetargeter


class CollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        objective_tests.ObjectiveTests.setUpClass();cls.model=objective_tests.ObjectiveTests.model

    def test_distance_gradients_including_penetration(self):
        a=DexHandRetargeter(self.model);c=a.collision;opt=a.retargeting.optimizer
        seen_penetration=False
        for seed in (2,7):
            q=np.zeros(c.robot.dof);q[opt.idx_pin2target]=np.random.default_rng(seed).uniform(.05,1.1,opt.opt_dof)
            d,j=c.distances(q,True);seen_penetration |= d.min()<0
            numerical=np.zeros_like(j);eps=1e-6
            for k in opt.idx_pin2target:
                dq=np.zeros_like(q);dq[k]=eps
                numerical[:,k]=(c.distances(q+dq)-c.distances(q-dq))/(2*eps)
            np.testing.assert_allclose(j[:,opt.idx_pin2target],numerical[:,opt.idx_pin2target],atol=2e-6,rtol=1e-3)
        self.assertTrue(seen_penetration)

    def test_guard_rejects_penetrating_command(self):
        a=DexHandRetargeter(self.model);c=a.collision
        zero=np.zeros(c.robot.dof)
        self.assertGreaterEqual(c.distances(zero).min(),0)
        q=zero.copy();q[a.retargeting.optimizer.idx_pin2target]=np.random.default_rng(2).uniform(.05,1.1,20)
        self.assertLess(c.distances(q).min(),0)
        accepted=c.accept(zero,q)
        self.assertGreaterEqual(c.distances(accepted).min(),-c.tolerance)
        self.assertFalse(np.allclose(q,accepted))

if __name__=='__main__':unittest.main()
