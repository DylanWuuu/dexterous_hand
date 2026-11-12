"""Numerical checks for the local objective and custom robot geometry."""
import sys,unittest
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dex_hand_retargeting import DexHandRetargeter,FINGER_LINKS,TIP_OFFSETS


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT/'generated/hand.xml').exists():
            from prepare_model import prepare
            prepare()
        cls.model=mujoco.MjModel.from_xml_path(str(ROOT/'generated/hand.xml'))

    def reference(self, adapter, angle=.35):
        opt=adapter.retargeting.optimizer;robot=opt.robot
        q=np.zeros(robot.dof);q[opt.idx_pin2target]=angle
        robot.compute_forward_kinematics(q)
        def positions(names):
            return np.array([robot.get_link_pose(robot.get_link_index(n))[:3,3].copy() for n in names])
        points=positions(['link_003']+[name for chain in FINGER_LINKS for name in chain])
        ref=positions(opt.task_link_names)-positions(opt.origin_link_names)
        return points,ref

    def test_full_objective_gradient_for_both_methods(self):
        for method in ('vector','dexpilot'):
            with self.subTest(method=method):
                adapter=DexHandRetargeter(self.model,method=method)
                points,ref=self.reference(adapter)
                adapter.objective.set_targets(points)
                opt=adapter.retargeting.optimizer
                x=np.full(opt.opt_dof,.2);last=np.full(opt.opt_dof,.4)
                function=opt.get_objective_function(ref,adapter.fixed,last)
                analytical=np.zeros_like(x);function(x,analytical)
                numerical=np.zeros_like(x);eps=1e-5
                for j in range(len(x)):
                    d=np.zeros_like(x);d[j]=eps
                    numerical[j]=(function(x+d,np.empty(0))-function(x-d,np.empty(0)))/(2*eps)
                np.testing.assert_allclose(analytical,numerical,rtol=1e-5,atol=1e-8)

    def test_reachable_direction_targets_and_zero_length_rejection(self):
        adapter=DexHandRetargeter(self.model)
        points,ref=self.reference(adapter)
        adapter.objective.set_targets(points)
        opt=adapter.retargeting.optimizer;x=np.full(opt.opt_dof,.35)
        function=opt.get_objective_function(ref,adapter.fixed,x)
        gradient=np.zeros_like(x)
        self.assertLess(function(x,gradient),1e-12)
        np.testing.assert_allclose(gradient,0,atol=1e-9)
        bad=points.copy();bad[2]=bad[1]
        with self.assertRaises(ValueError):adapter.objective.set_targets(bad)

    def test_pinocchio_mujoco_same_geometry(self):
        adapter=DexHandRetargeter(self.model);robot=adapter.retargeting.optimizer.robot
        names=['link_003']+[name for chain in FINGER_LINKS for name in chain]
        data=mujoco.MjData(self.model)
        for seed in range(5):
            q=np.zeros(self.model.nq) if seed==0 else np.random.default_rng(seed).uniform(self.model.jnt_range[:,0],self.model.jnt_range[:,1])
            robot.compute_forward_kinematics(q[adapter.sim_indices]);data.qpos[:]=q
            mujoco.mj_forward(self.model,data)
            pin=np.array([robot.get_link_pose(robot.get_link_index(n))[:3,3].copy() for n in names])
            sim=[]
            for name in names:
                if name.startswith('dex_'):
                    parent,offset=TIP_OFFSETS[name[4:-4]];i=self.model.body(parent).id
                    sim.append(data.xpos[i]+data.xmat[i].reshape(3,3)@offset)
                else:sim.append(data.xpos[self.model.body(name).id].copy())
            sim=np.array(sim)
            np.testing.assert_allclose(np.linalg.norm(pin[:,None]-pin[None,:],axis=2),
                                       np.linalg.norm(sim[:,None]-sim[None,:],axis=2),atol=2e-7,rtol=0)

    def test_reset_starts_at_model_zero(self):
        adapter=DexHandRetargeter(self.model)
        adapter.retargeting.last_qpos[:]=.7
        adapter.sequence=99;adapter.reset()
        np.testing.assert_array_equal(adapter.retargeting.last_qpos,0)
        self.assertIsNone(adapter.sequence)
        self.assertFalse(adapter.retargeting.filter.is_init)


if __name__=='__main__':unittest.main()
