"""Compile the supplied URDF without changing the original files."""
from pathlib import Path
import xml.etree.ElementTree as ET
import struct
import json
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'robot_urdf/robot/robot.urdf'
BUILD = ROOT / 'generated'

def prepare():
    BUILD.mkdir(exist_ok=True)
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    repaired = []
    paths = {}
    for node in root.findall('.//mesh'):
        name = node.get('filename')
        if name not in paths:
            path = SOURCE.parent / name
            raw = path.read_bytes()
            count = struct.unpack_from('<I', raw, 80)[0]
            dtype = np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
            records = np.frombuffer(raw, dtype=dtype, count=count, offset=84)
            vertices = records['vertices'].reshape(-1,3).astype(float)
            _, singular, axes = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)
            if singular[-1] < max(singular[0]*1e-7, 1e-8):
                # CAD surface-only parts have zero thickness: add a microscopic
                # tetrahedron so MuJoCo can construct a 3D convex hull.
                a,b,c = records['vertices'][0].astype(float)
                tip = (a+b+c)/3 + axes[-1]*0.01 # STL is millimetres: 10 micrometres
                extra = []
                for x,y in [(a,b),(b,c),(c,a)]:
                    extra.append(struct.pack('<12fH',0,0,0,*x,*y,*tip,0))
                target = BUILD / path.name
                target.write_bytes(raw[:80]+struct.pack('<I',count+3)+raw[84:84+count*50]+b''.join(extra))
                path = target
                repaired.append(name)
            paths[name] = str(path.resolve()).replace('\\','/')
        node.set('filename', paths[name])
    ET.SubElement(ET.SubElement(root,'mujoco'),'compiler',discardvisual='false',fusestatic='false',balanceinertia='true')
    spec = mujoco.MjSpec.from_string(ET.tostring(root,encoding='unicode'))
    for mesh in spec.meshes:
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    model = spec.compile()
    (BUILD/'hand.xml').write_text(spec.to_xml(),encoding='utf-8')
    xml = ET.parse(BUILD/'hand.xml')
    r = xml.getroot()
    ET.SubElement(r,'option',timestep='0.002',gravity='0 0 0',integrator='implicitfast')
    default = r.find('default')
    if default is None: default = ET.SubElement(r,'default')
    ET.SubElement(default,'joint',damping='0.03',armature='0.001')
    # CAD shells overlap; disable robot self contact for joint commissioning.
    # Contacts with grasped objects need separately validated collision geometry.
    for geom in r.findall('.//geom'):
        geom.set('contype','0'); geom.set('conaffinity','0')
        geom.set('rgba','0.65 0.72 0.82 1')
    act = ET.SubElement(r,'actuator')
    for joint in r.findall('.//joint'):
        if joint.get('name'):
            ET.SubElement(act,'position',name=joint.get('name')+'_servo',joint=joint.get('name'),kp='4',kv='0.15',forcerange='-1 1',ctrlrange=joint.get('range','-1.57 1.57'))
    ET.indent(xml)
    xml.write(BUILD/'hand.xml',encoding='utf-8',xml_declaration=True)
    final = mujoco.MjModel.from_xml_path(str(BUILD/'hand.xml'))
    report = dict(source=str(SOURCE),joints=final.nq,geoms=final.ngeom,actuators=final.nu,thin_mesh_repairs=repaired,self_collision=False,gravity=False)
    (BUILD/'model_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return final

if __name__ == '__main__': prepare()
