"""Official Qnbot calibrated Skeleton -> upstream dex-retargeting -> MuJoCo."""
import argparse,json,os,socket,subprocess,sys,time,traceback,atexit,signal,threading,select
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import numpy as np
import mujoco
import mujoco.viewer
from dex_hand_retargeting import DexHandRetargeter,FINGERS
from pose_mapping import POSES, SAMPLE_COUNT, PoseMapping, mapping_path
from project_runtime import InstanceLock, detect_glove_port

ROOT=Path(__file__).resolve().parent
CAPTURE_HINTS = {
    'close': '食指到小指自然伸直并靠拢，拇指自然贴向食指侧，保持手腕平稳。',
    'mcp90': '食指到小指的指根弯曲约 90°，中节和末节保持伸直。',
    'mcp90_pip90': '自然握拳，四指向掌心弯曲。',
}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',help='指定串口，默认由 Qnbot SDK 自动查找左手手套');ap.add_argument('--side',choices=['left','right'],default='left');ap.add_argument('--seconds',type=float,default=0);ap.add_argument('--calibrate',action='store_true');ap.add_argument('--no-collision',action='store_true',help='关闭实验性碰撞约束进行对照');ap.add_argument('--retargeting',choices=['vector','dexpilot'],default='vector');args=ap.parse_args()
    args.port = args.port or detect_glove_port(args.side)
    owner=InstanceLock('ui',args.port);atexit.register(owner.close)
    stopping=threading.Event()
    signal.signal(signal.SIGINT,lambda *_:stopping.set());signal.signal(signal.SIGTERM,lambda *_:stopping.set())
    model_path = ROOT / 'generated/hand.xml'
    if not model_path.exists():
        from prepare_model import prepare
        prepare()
    model=mujoco.MjModel.from_xml_path(str(model_path));data=mujoco.MjData(model)
    retargeter=DexHandRetargeter(model,args.side,args.retargeting,collision=not args.no_collision)
    collision_label=" + 简化防碰撞" if retargeter.collision else ""
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.bind(('127.0.0.1',0));sock.setblocking(False)
    log=open(ROOT/'generated/qnbot_bridge.log','w',encoding='utf-8')
    env=os.environ.copy();env.setdefault('QNBOT_HOME',str(ROOT/'qnbot_data'))
    def launch_bridge(force=False):
        command=[sys.executable,str(ROOT/'qnbot_calibrated_bridge.py'),'--port',args.port,'--side',args.side,'--udp',str(sock.getsockname()[1]),'--parent-pid',str(os.getpid())]
        if force:command.append('--force')
        return subprocess.Popen(command,stdout=log,stderr=log,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    bridge=launch_bridge(args.calibrate)
    bridge_address=[None];closed=[False];connection_error=[None];bridge_pid=[bridge.pid]
    def release_resources():
        if closed[0]:return
        closed[0]=True
        # Ask the SDK owner to leave its update loop and execute its finally block.
        deadline=time.monotonic()+8.
        while bridge.poll() is None and time.monotonic()<deadline:
            try:
                ready,_,_=select.select([sock],[],[],.15)
                if ready:
                    while True:
                        _,addr=sock.recvfrom(65535);bridge_address[0]=addr
                if bridge_address[0]:sock.sendto(b'{"action":"shutdown"}',bridge_address[0])
            except BlockingIOError:
                if bridge_address[0]:
                    try:sock.sendto(b'{"action":"shutdown"}',bridge_address[0])
                    except OSError:pass
            except OSError:pass
        if bridge.poll() is None:
            print(f'WARNING: SDK owner PID {bridge.pid} has not acknowledged close; it will close when parent exits. No second connection will be opened.',file=sys.stderr,flush=True)
        else:print(f'SDK owner exited rc={bridge.returncode}',flush=True)
        sock.close();log.close();owner.close()
    atexit.register(release_resources)
    ui=tk.Tk();ui.title('Qnbot 官方标定 → dex-retargeting');ui.geometry('760x610')
    ttk.Label(ui,text=f'Qnbot SDK | {args.side} | {args.port}',font=('Microsoft YaHei',15)).pack(pady=12)
    state=tk.StringVar(value='正在连接手套…');stats=tk.StringVar();distance=tk.StringVar();caltext=tk.StringVar(value='正在检查 Qnbot 官方标定…')
    ttk.Label(ui,textvariable=state,wraplength=660,font=('Microsoft YaHei',11)).pack(pady=10)
    ttk.Label(ui,text=f'现有官方标定 → Skeleton → 21 关键点 → Dex {args.retargeting} + 指节方向{collision_label} → 机器人',wraplength=700).pack(pady=6)
    ttk.Label(ui,textvariable=caltext,wraplength=700,font=('Microsoft YaHei',11)).pack(pady=5)
    pending=[None];confirmed=set();calibrating=[False]
    def confirm_stage():
        request=pending[0]
        if request and bridge_address[0] and request not in confirmed:
            sock.sendto(json.dumps({'action':'confirm','request_id':request}).encode(),bridge_address[0]);confirmed.add(request);confirm_button.configure(state='disabled')
    confirm_button=ttk.Button(ui,text='姿势稳定，确认本步',command=confirm_stage,state='disabled');confirm_button.pack(pady=4)
    review_buttons = ttk.Frame(ui)
    mapping_text = tk.StringVar()
    ttk.Label(review_buttons, textvariable=mapping_text, wraplength=700).pack(pady=4)
    controls = ttk.Frame(review_buttons)
    controls.pack()
    reviewing = False
    captures = {}
    sampling = None
    sample_sequence = None
    mapping_error = None

    def current_mapping_path():
        return mapping_path(Path(env['QNBOT_HOME']), model_path, args.retargeting,
                            collision=retargeter.collision is not None)

    def prompt_pose():
        index = len(captures)
        mapping_text.set(f'参考动作 {index + 1}/{len(POSES)} · {POSES[index][1]}。保持后点击采样。')
        sample_button.configure(state='normal')

    def reset_sampling():
        nonlocal sampling, sample_sequence
        captures.clear()
        sampling = None
        sample_sequence = None
        retargeter.pose_mapping = None
        retain_button.configure(state='disabled')
        redo_button.configure(state='disabled')
        prompt_pose()

    def begin_review():
        nonlocal reviewing
        if not reviewing:
            reviewing = True
            reset_sampling()
            reject_button.configure(state='normal')
            review_buttons.pack(pady=4)

    def sample_pose():
        nonlocal sampling, sample_sequence
        sampling = POSES[len(captures)]
        sample_sequence = retargeter.sequence
        sample_button.configure(state='disabled')
        redo_button.configure(state='normal')
        mapping_text.set('正在等待有效骨骼，请保持当前姿势。')

    def collect_pose(packet):
        nonlocal sampling, sample_sequence, mapping_error
        sequence = packet.get('skeleton_sequence')
        if (not sampling or not packet.get('calibration_review')
                or not packet.get('skeleton_ready') or not packet.get('skeleton_nodes')
                or sequence is None
                or sequence == sample_sequence):
            return
        sample_sequence = sequence
        key, name = sampling
        frames = captures.setdefault(key, [])
        frames.append(packet['skeleton_nodes'])
        mapping_text.set(f'{name} · {len(frames)}/{SAMPLE_COUNT} 帧')
        if len(frames) < SAMPLE_COUNT:
            return
        sampling = None
        redo_button.configure(state='normal')
        if len(captures) < len(POSES):
            prompt_pose()
            return
        try:
            retargeter.fit_pose_mapping(captures)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            mapping_text.set(f'映射拟合失败。{exc}')
            return
        mapping_error = None
        mapping_text.set('映射已应用。请试张开、握拳、单指弯曲和捏合，再决定是否保留。')
        retain_button.configure(state='normal')

    def review_calibration(action):
        if action == 'accept_calibration':
            try:
                retargeter.pose_mapping.save(current_mapping_path())
            except OSError as exc:
                mapping_text.set(f'无法保存映射。{exc}')
                return
        if bridge_address[0]:
            sock.sendto(json.dumps({'action': action}).encode(), bridge_address[0])
            for button in (sample_button, redo_button, retain_button, reject_button):
                button.configure(state='disabled')

    sample_button = ttk.Button(controls, text='采样当前姿态', command=sample_pose)
    redo_button = ttk.Button(controls, text='重采参考姿态', command=reset_sampling)
    retain_button = ttk.Button(controls, text='保留新标定和映射',
                               command=lambda: review_calibration('accept_calibration'))
    reject_button = ttk.Button(controls, text='恢复旧标定',
                               command=lambda: review_calibration('reject_calibration'))
    for button in (sample_button, redo_button, retain_button, reject_button):
        button.pack(side='left', padx=4)
    markers=tk.BooleanVar(value=True);labels=tk.BooleanVar(value=False)
    ttk.Checkbutton(ui,text='显示手套指尖',variable=markers).pack()
    ttk.Label(ui,textvariable=distance).pack(pady=8);ttk.Label(ui,textvariable=stats).pack(pady=6)
    ttk.Label(ui,text='优化 20 个手指关节，手腕固定；仅仿真，物理接触力响应关闭。').pack()
    alive=True
    waiting_for_calibrated_frame = args.calibrate
    def close():
        nonlocal alive
        alive=False
    ui.protocol('WM_DELETE_WINDOW',close)
    started=time.monotonic();last_sequence=-1;last_joint_sequence=None;last_data=0.;last_log=0.;last_stats=0.;bad=[];target=np.zeros(model.nq);human=None;count=0;input_source=None;mode_error=None;angle_updates=0
    def restart_calibration():
        nonlocal waiting_for_calibrated_frame, mode_error
        if bridge.poll() is not None or not bridge_address[0]:
            caltext.set('SDK 未连接。请解决串口错误后重新启动程序。');return
        if calibrating[0]:caltext.set('官方采集或标定已在进行，请完成当前步骤。');return
        sock.sendto(b'{"action":"recalibrate"}',bridge_address[0])
        waiting_for_calibrated_frame = True
        calibrating[0] = True
        mode_error = None
        pending[0]=None;confirmed.clear();confirm_button.configure(state='disabled')
        caltext.set('已请求同一 SDK 开始官方八步采集。')
    ttk.Button(ui,text='重新进行官方手势标定',command=restart_calibration).pack(pady=3)
    try:
        with mujoco.viewer.launch_passive(model,data,show_left_ui=False,show_right_ui=False) as viewer:
            viewer.cam.lookat[:]=[0,0,.17];viewer.cam.distance=.47;viewer.cam.azimuth=135;viewer.cam.elevation=-20
            viewer.opt.label=mujoco.mjtLabel.mjLABEL_NONE;viewer.opt.geomgroup[0]=False;viewer.opt.geomgroup[1]=True
            print('QNBOT SDK SIM READY',flush=True)
            physics_tick=time.monotonic()
            while alive and not stopping.is_set() and viewer.is_running():
                tick=time.monotonic();ui.update()
                if args.seconds and tick-started>args.seconds:break
                packet=None
                try:
                    while True:
                        raw,addr=sock.recvfrom(65535);message=json.loads(raw);bridge_address[0]=addr
                        if message.get('type')=='frame':
                            packet=message
                        elif message.get('type')=='connected':
                            bridge_pid[0]=message['pid'];connection_error[0]=None
                            try:
                                path = current_mapping_path()
                                if path.is_file():
                                    retargeter.pose_mapping = PoseMapping.load(path)
                            except (OSError, ValueError, TypeError) as exc:
                                mapping_error = f'无法加载 Dex 映射。{exc}'
                        elif message.get('type')=='calibration_started':
                            calibrating[0] = True
                            waiting_for_calibrated_frame = True
                            mode_error = None
                            packet = None
                        elif message.get('type')=='error':
                            review_buttons.pack_forget()
                            caltext.set('SDK: '+message.get('message','未知错误'))
                            if message.get('fatal'):connection_error[0]=message['message']
                        elif message.get('type')=='status':
                            cap=message.get('capture');cal=message.get('calibration');pending[0]=None
                            review_scale = message.get('calibration_review_scale')
                            calibrating[0]=bool(review_scale is not None or (cap and cap['session_state']=='running') or (cal and cal['state'] in ('pending','running')))
                            if review_scale is None and reviewing:
                                reviewing = False
                                packet = None
                                review_buttons.pack_forget()
                            if review_scale is not None:
                                caltext.set('官方标定已完成。请采样七个参考姿态，拟合本次 Dex 映射。'
                                            if reviewing else '官方标定已完成，正在等待本次标定的新骨骼帧。')
                            elif cap and cap['session_state']=='failed':caltext.set('官方采集失败：'+str(cap['failure']))
                            elif cal and cal['state']=='failed':caltext.set('官方标定失败：'+str(cal['failure']))
                            elif cap and cap['session_state']=='running':
                                hint = CAPTURE_HINTS.get(cap['stage_id'], '')
                                caltext.set(f"{cap['prompt']} {hint} 采样 {cap['collected']}/{cap['required']}")
                                if cap['state']=='awaiting_confirmation':pending[0]=cap['request_id']
                            elif cal and cal['state'] in ('pending','running'):caltext.set('SDK 正在计算官方标定。指尖预览继续，计算完成后可试用新标定。')
                            elif input_source=='qnbot_calibrated_skeleton':
                                mode = '已应用本次标定的 Dex 映射。' if retargeter.pose_mapping else '使用原始 Dex 目标。'
                                caltext.set('官方标定已就绪。' + (mapping_error or mode))
                            confirm_button.configure(state='normal' if pending[0] and pending[0] not in confirmed else 'disabled')
                except BlockingIOError:pass
                if packet and packet['sequence']!=last_sequence:
                    last_sequence=packet['sequence'];count+=1;last_data=tick
                    input_source=packet['source']
                    bad=[f for f in FINGERS if f not in packet['positions'] or not packet['valid'].get(f,False) or packet['held'].get(f,False) or not np.isfinite(packet['positions'][f]).all()]
                    if not bad:human=np.array([np.zeros(3)]+[packet['positions'][f] for f in FINGERS])
                    if (not bad and input_source == 'qnbot_calibrated_skeleton'
                            and packet.get('skeleton_ready')
                            and (not waiting_for_calibrated_frame or packet.get('calibration_review'))):
                        if packet.get('calibration_review') and not reviewing:
                            begin_review()
                            retargeter.reset()
                            # Start the simulator and collision checks from the same zero pose.
                            mujoco.mj_resetData(model, data)
                            mujoco.mj_forward(model, data)
                            target[:] = 0.
                            physics_tick = tick
                            last_joint_sequence = None
                        waiting_for_calibrated_frame = False
                        collect_pose(packet)
                        try:
                            target=retargeter.retarget(packet)
                            mode_error=None
                            if retargeter.sequence!=last_joint_sequence:
                                last_joint_sequence=retargeter.sequence;angle_updates+=1
                        except (ValueError,RuntimeError) as exc:
                            mode_error=str(exc);target[:]=data.qpos
                fresh=tick-last_data<.5 and not bad
                if not fresh:target[:]=data.qpos
                data.ctrl[:]=np.clip(target,model.jnt_range[:,0],model.jnt_range[:,1])
                # Advance simulation by elapsed wall time, not a fixed 10 ms
                # per optimization frame (which slowed physics as solving grew).
                steps=max(0,min(50,int((tick-physics_tick)/model.opt.timestep)))
                if waiting_for_calibrated_frame:
                    physics_tick = tick
                else:
                    for _ in range(steps):mujoco.mj_step(model,data)
                    physics_tick+=steps*model.opt.timestep
                if tick-physics_tick>.1:physics_tick=tick
                with viewer.lock():
                    viewer.opt.label=mujoco.mjtLabel.mjLABEL_GEOM if labels.get() else mujoco.mjtLabel.mjLABEL_NONE
                    viewer.user_scn.ngeom=0
                    raw_preview = input_source == 'qnbot_raw_calibration_preview'
                    if (raw_preview or (markers.get() and not waiting_for_calibrated_frame)) and fresh and human is not None:
                        # Input-only display in its own fixed wrist-local axes.
                        points=human@np.array([[0,0,1],[0,1,0],[-1,0,0]])+np.array([.2,0,.12])
                        for i in range(5):
                            geom=viewer.user_scn.geoms[i];color=np.array([(i%3)/3+.2,((i+1)%3)/3+.2,((i+2)%3)/3+.2,1.])
                            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.004),points[i+1],np.eye(3).ravel(),color);viewer.user_scn.ngeom+=1
                viewer.sync()
                if connection_error[0]:state.set(connection_error[0])
                elif bridge.poll() is not None:state.set('手套连接已停止，不自动重试；检查端口后重新启动。详见 generated/qnbot_bridge.log')
                elif waiting_for_calibrated_frame:state.set('Qnbot 官方采集或标定中，等待本次标定的新骨骼。机器人保持姿态。')
                elif not fresh or input_source != 'qnbot_calibrated_skeleton':state.set('等待标定后的有效手套数据 / 手指无效：'+', '.join(bad))
                elif mode_error:state.set(mode_error)
                elif retargeter.collision and retargeter.collision.last_held:state.set('骨骼数据有效；防碰撞限制了当前运动，机器人保持上一安全姿态。')
                else:state.set(f'Dex {args.retargeting} + 指节方向{collision_label} | 单帧 {retargeter.last_ms:.1f} ms')
                if tick-last_stats>.2:
                    distance.set('输入：'+('SDK 原始指尖预览' if waiting_for_calibrated_frame
                                             else '官方标定后的 Skeleton' if input_source == 'qnbot_calibrated_skeleton'
                                             else '等待官方标定后的实时 Skeleton'))
                    stats.set(f'帧 {last_sequence} | 重定向 {angle_updates} 次'+(f' | 防碰撞缩步 {retargeter.collision.safe_steps} / 保持 {retargeter.collision.held_frames} 帧' if retargeter.collision else ''));last_stats=tick
                if tick-last_log>1:
                    report={'mode':'dex-retargeting '+args.retargeting,'sdk_owner_pid':bridge_pid[0],'connection_error':connection_error[0],'input':input_source,'frame_sequence':last_sequence,'skeleton_sequence':last_joint_sequence,'updates':count,'retarget_updates':angle_updates,'retarget_ms':round(retargeter.last_ms,2),'collision_enabled':retargeter.collision is not None,'collision_held_frames':retargeter.collision.held_frames if retargeter.collision else 0,'collision_holding':retargeter.collision.last_held if retargeter.collision else False,'collision_step_scale':retargeter.collision.last_scale if retargeter.collision else 1.,'data_fresh':fresh,'calibrating':calibrating[0],'retargeting_paused':waiting_for_calibrated_frame,'invalid_fingers':bad,'mode_error':mode_error,'joint_motion_rad':float(np.max(np.abs(data.qpos))),'geom_labels':labels.get(),'colored_fingertips':markers.get(),'elapsed_s':round(tick-started,1)}
                    (ROOT/'generated/live_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');last_log=tick
                time.sleep(max(0,.01-(time.monotonic()-tick)))
    finally:
        release_resources();ui.destroy()
    print('QNBOT SDK SIM EXIT',count,angle_updates,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
