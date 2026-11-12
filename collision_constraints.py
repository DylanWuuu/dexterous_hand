"""Experimental capsule/box self-collision constraints; not full CAD contact."""
import numpy as np
import mujoco


class HandCollisionConstraints:
    tolerance=1e-6  # metres, checked independently after solving
    clearance=5e-5  # numerical reserve for interpolation, not a large pinch gap

    def __init__(self, optimizer, chains):
        self.optimizer=optimizer;self.robot=optimizer.robot
        self.robot.compute_forward_kinematics(np.zeros(self.robot.dof))
        geoms=[];self.link_ids=[]
        for finger,chain in enumerate(chains):
            for segment,(start,end) in enumerate(zip(chain[:-1],chain[1:])):
                i=self.robot.get_link_index(start);j=self.robot.get_link_index(end)
                a=self.robot.get_link_pose(i).copy();b=self.robot.get_link_pose(j)[:3,3].copy()
                v=a[:3,:3].T@(b-a[:3,3]);v[2]=0.;length=np.linalg.norm(v);u=v/length
                radius=.0085 if finger==0 else .0075
                # Rounded ends terminate at the semantic joint/tip, rather than
                # protruding beyond the fingertip by a whole capsule radius.
                p=u*radius;q=v-u*radius
                z=((.0073,.0054,.0056),(-.0073,.0044,.0066),(.0062,.0047,.0040),(-.0073,-.0041,-.0046),(.0059,-.0058,-.0056))[finger][segment]
                p[2]=q[2]=z  # CAD axial centre, not the displaced joint axis
                geoms.append(f'<geom type="capsule" size="{radius}" fromto="'+ ' '.join(map(str,np.r_[p,q]))+'"/>')
                self.link_ids.append(i)
        # AABB of link_003 CAD in its own frame; excludes other bodies.
        # Capsules approximate phalanges, not every CAD shell or joint protrusion.
        geoms.append('<geom type="box" pos=".0413 -.00045 .0171" size=".0473 .03895 .0269"/>')
        self.link_ids.append(self.robot.get_link_index('link_003'))
        self.model=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+''.join(geoms)+'</worldbody></mujoco>')
        self.data=mujoco.MjData(self.model)
        self.local_pos=self.model.geom_pos.copy();self.local_rot=[]
        for quat in self.model.geom_quat:
            matrix=np.zeros(9);mujoco.mju_quat2Mat(matrix,quat);self.local_rot.append(matrix.reshape(3,3))
        self.pairs=[(a,b) for a in range(15) for b in range(a+1,15) if a//3!=b//3 or b-a>1]
        # Proximal segments join the palm and are intentionally excluded there.
        self.pairs += [(a,15) for a in range(15) if a%3!=0]
        self.reset()
        optimizer.opt.add_inequality_mconstraint(self.constraint,np.full(len(self.pairs),1e-5))
        if self.distances(np.zeros(self.robot.dof)).min() < -self.tolerance:
            raise ValueError('简化碰撞体在模型零位重叠，需要调整碰撞几何')

    def reset(self):
        self.last_x = None
        self.safe_steps = 0
        self.held_frames = 0
        self.last_held = False
        self.last_scale = 1.

    def distances(self, q, derivatives=False):
        self.robot.compute_forward_kinematics(q)
        poses=[self.robot.get_link_pose(i).copy() for i in self.link_ids]
        for g,pose in enumerate(poses):
            self.data.geom_xpos[g]=pose[:3,3]+pose[:3,:3]@self.local_pos[g]
            self.data.geom_xmat[g]=(pose[:3,:3]@self.local_rot[g]).ravel()
        jac=[]
        if derivatives:
            jac=[pose[:3,:3]@self.robot.compute_single_link_local_jacobian(q,i).reshape(2,3,-1)
                 for i,pose in zip(self.link_ids,poses)]
        distances=np.empty(len(self.pairs));witnesses=np.zeros((len(self.pairs),6))
        for k,(a,b) in enumerate(self.pairs):
            distances[k]=mujoco.mj_geomDistance(self.model,self.data,a,b,1.,witnesses[k])
        if not derivatives:return distances
        a,b=np.array(self.pairs).T;jac=np.array(jac)
        normals=np.divide(witnesses[:,3:]-witnesses[:,:3],distances[:,None],
                          out=np.zeros((len(distances),3)),where=np.abs(distances[:,None])>1e-10)
        origins=np.array([p[:3,3] for p in poses])
        def velocities(ids,points):
            return jac[ids,0]+np.cross(jac[ids,1].transpose(0,2,1),(points-origins[ids])[:,None,:]).transpose(0,2,1)
        gradients=np.einsum('ni,nij->nj',normals,velocities(b,witnesses[:,3:])-velocities(a,witnesses[:,:3]))
        return distances,gradients

    def constraint(self, result, x, grad):
        opt=self.optimizer;q=np.zeros(self.robot.dof);q[opt.idx_pin2target]=x
        if self.last_x is None or not np.array_equal(x,self.last_x):
            self.last_distances,self.last_jac=self.distances(q,True);self.last_x=x.copy()
        result[:]=(self.clearance-self.last_distances)/.01
        if grad.size:grad[:]=-self.last_jac[:,opt.idx_pin2target]/.01

    def accept(self, previous, candidate):
        # Check the filtered command AND several points along the transition.
        for step in range(9):
            target=previous+(candidate-previous)*(0.5**step)
            if all(self.distances(previous+(target-previous)*t).min()>=-self.tolerance for t in (.25,.5,.75,1.)):
                self.safe_steps+=step>0
                self.last_held=False;self.last_scale=0.5**step
                return target
        self.held_frames+=1
        self.last_held=True;self.last_scale=0.
        return previous.copy()
