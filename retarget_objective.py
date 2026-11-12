"""Local corrections around upstream dex-retargeting; installed package stays intact."""
import numpy as np


class FingerDirectionObjective:
    def __init__(self, optimizer, finger_links, weight=.3):
        self.optimizer=optimizer
        self.weight=weight
        self.link_ids=[optimizer.robot.get_link_index(name) for chain in finger_links for name in chain]
        self.origins=np.array([i for start in range(0,20,4) for i in range(start,start+3)])
        self.tasks=self.origins+1
        self.targets=None
        self.upstream_objective=optimizer.get_objective_function
        optimizer.get_objective_function=self.get_objective_function

    def set_targets(self, points):
        # Human indices 1..20 are five chains of four landmarks each.
        vectors=points[self.tasks+1]-points[self.origins+1]
        lengths=np.linalg.norm(vectors,axis=1,keepdims=True)
        if np.any(lengths<1e-8):raise ValueError('SDK 指节长度退化，不能计算方向目标')
        self.targets=vectors/lengths

    def get_objective_function(self, ref_value, fixed_qpos, last_qpos):
        opt=self.optimizer;robot=opt.robot
        base=self.upstream_objective(ref_value,fixed_qpos,last_qpos)
        targets=None if self.targets is None else self.targets.copy()
        qpos=np.zeros(robot.dof);qpos[opt.idx_pin2fixed]=fixed_qpos

        def objective(x, grad):
            # Upstream already adds this regularizer's derivative, but omits
            # its value. Add only the missing value, never its gradient twice.
            delta=x-last_qpos
            result=base(x,grad)+opt.norm_delta*np.dot(delta,delta)
            if targets is None:return result
            # base() has evaluated FK at x; reuse that same state.
            poses=[robot.get_link_pose(i).copy() for i in self.link_ids]
            positions=np.array([pose[:3,3] for pose in poses])
            vectors=positions[self.tasks]-positions[self.origins]
            lengths=np.linalg.norm(vectors,axis=1,keepdims=True)
            directions=vectors/lengths
            residual=directions-targets
            result+=.5*self.weight*np.mean(np.sum(residual**2,axis=1))
            if grad.size:
                qpos[opt.idx_pin2target]=x
                if opt.adaptor is not None:qpos[:]=opt.adaptor.forward_qpos(qpos)
                jacobians=np.array([pose[:3,:3]@robot.compute_single_link_local_jacobian(qpos,i)[:3]
                                    for i,pose in zip(self.link_ids,poses)])
                if opt.adaptor is not None:jacobians=opt.adaptor.backward_jacobian(jacobians)
                else:jacobians=jacobians[...,opt.idx_pin2target]
                # Derivative of v/|v|: (I-u*u.T)/|v|.
                dv=self.weight/len(vectors)*(residual-directions*np.sum(residual*directions,axis=1,keepdims=True))/lengths
                grad+=np.einsum('ni,nij->j',dv,jacobians[self.tasks]-jacobians[self.origins])
            return float(result)

        return objective
