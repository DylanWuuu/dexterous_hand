"""One SDK owner for capture, Calibration, Skeleton and joint-angle reads."""
import argparse,json,os,signal,socket,sys,time,threading
from contextlib import ExitStack
from pathlib import Path
from calibration_guard import CalibrationGuard, CalibrationRejected
from qnbot_sdk import Sdk,Side,SerialConnection,TargetName,AlgorithmsConfig,CaptureConfig,CaptureInteractionMode
from qnbot_sdk.glove import GloveConfig,CaptureStageState,CalibrationMode
from project_runtime import InstanceLock,ParentWatch,check_port_available

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',required=True);ap.add_argument('--side',required=True);ap.add_argument('--udp',type=int,required=True);ap.add_argument('--parent-pid',type=int);ap.add_argument('--force',action='store_true');args=ap.parse_args()
    stop=threading.Event()
    def request_stop(*_):stop.set()
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.bind(('127.0.0.1',0));sock.setblocking(False)
    # A Windows UDP ICMP error means the UI endpoint vanished, not a serial error.
    if hasattr(socket,'SIO_UDP_CONNRESET'):sock.ioctl(socket.SIO_UDP_CONNRESET,False)
    sdk=glove=capture_session=calibration_session=None
    sdk_creations=0
    guard = CalibrationGuard(os.environ.get('QNBOT_HOME', Path.home() / '.qnbot'))
    calibration_validated = False
    review_scale = None
    def send(msg):sock.sendto(json.dumps(msg,ensure_ascii=False).encode('utf-8'),('127.0.0.1',args.udp))
    def close_sessions():
        nonlocal capture_session,calibration_session
        for session in (calibration_session,capture_session):
            if session is not None:
                try:session.close()
                except Exception as exc:print(f'Session close: {exc}',file=sys.stderr,flush=True)
        capture_session=calibration_session=None
    try:
        send({'type':'hello','pid':os.getpid()})
        with ExitStack() as owned:
            parent=ParentWatch(args.parent_pid);owned.callback(parent.close)
            owned.enter_context(InstanceLock('device',args.port))
            check_port_available(args.port)
            # An interrupted calibration may have saved data before validation.
            guard.rollback()
            try:
                config=AlgorithmsConfig(capture=CaptureConfig(interaction=CaptureInteractionMode.EXTERNAL))
                sdk=Sdk(devices=(GloveConfig(side=Side(args.side),connection=SerialConnection(port=args.port)),),algorithms=config);sdk_creations+=1
                glove=sdk.glove();dev=glove.device();raw=dev.pose();skeleton=dev.skeleton();pose=skeleton.pose();angles=skeleton.joint_angles()
                progress=dev.capture_progress();calibration=dev.calibration_progress(name=TargetName.SKELETON);control=dev.capture_control()
                # No Python SDK reader thread is used; one manual update loop owns it.
                reuse_ready=dev.calibration_status(name=TargetName.SKELETON).state.value=='ready'
                device_started = reuse_ready
                if reuse_ready:
                    glove.start()
                last_key=None;last_send=0.;last_status=0.;frame_id=0;confirmed=set()
                last_angle_sequence=None;last_angle_update=float('-inf')
                last_skeleton_sequence=None;last_skeleton_update=float('-inf');skeleton_calibrated=reuse_ready
                capture_requested=args.force;capture_requested_at=time.monotonic()
                calibration_finished_sequence = None
                def begin_capture():
                    nonlocal capture_session, device_started
                    if capture_session and capture_session.snapshot().state.value=='running':raise RuntimeError('官方采集已在进行，请完成当前步骤')
                    if calibration_session and calibration_session.snapshot().state.value=='running':raise RuntimeError('官方标定计算中，请等待完成')
                    guard.begin()
                    close_sessions()
                    if device_started:
                        dev.stop()
                        device_started = False
                    dev.start(CalibrationMode.MANUAL)
                    device_started = True
                    capture_session=dev.start_capture(package_ids=(),operator_id='default',force=True)
                    confirmed.clear()
                    print(f'CAPTURE START sdk_creations={sdk_creations} pid={os.getpid()}',flush=True)
                    send({'type':'calibration_started','pid':os.getpid(),'sdk_creations':sdk_creations})
                send({'type':'connected','pid':os.getpid(),'sdk_creations':sdk_creations})
                if not reuse_ready:
                    begin_capture()
                    capture_requested = False
                while not stop.is_set() and parent.alive():
                    glove.update();now=time.monotonic();current=progress.latest()
                    try:
                        while True:
                            payload,addr=sock.recvfrom(4096)
                            if addr!=('127.0.0.1',args.udp):continue
                            message=json.loads(payload);action=message.get('action')
                            if action=='shutdown':stop.set();break
                            if action == 'accept_calibration' and review_scale is not None:
                                guard.commit()
                                review_scale = None
                            elif action == 'reject_calibration' and review_scale is not None:
                                raise CalibrationRejected('已放弃新标定')
                            elif action == 'recalibrate' and review_scale is None:
                                capture_requested=True;capture_requested_at=now
                            elif action=='confirm':
                                request=message.get('request_id')
                                if capture_session:
                                    snap=capture_session.snapshot();expected=snap.request_id
                                else:expected=current.value.stage.request_id if current else None
                                if request and request==expected and request not in confirmed:
                                    if capture_session:capture_session.confirm(request)
                                    elif current and current.value.stage.state==CaptureStageState.AWAITING_CONFIRMATION:control.confirm(request)
                                    confirmed.add(request)
                    except BlockingIOError:pass
                    # Let the existing official Skeleton initialize before changing mode.
                    # Otherwise immediate --force can stop it before its first output.
                    if capture_requested and (not reuse_ready or angles.latest() is not None or now-capture_requested_at>=5.):
                        capture_requested=False
                        try:begin_capture()
                        except Exception as exc:send({'type':'error','message':str(exc),'fatal':False})
                    # Official capture and Calibration sessions share this exact device/SDK.
                    if capture_session and capture_session.snapshot().state.value=='completed' and calibration_session is None:
                        calibration_validated = False
                        calibration_session=dev.start_calibration(completed_capture_session=capture_session,targets=(TargetName.SKELETON,))
                        print(f'CALIBRATION START sdk_creations={sdk_creations} pid={os.getpid()}',flush=True)
                    if calibration_session and not calibration_validated and calibration_session.snapshot().state.value == 'completed':
                        results = calibration_session.result() or ()
                        review_scale = guard.validate(result.record_id for result in results)
                        calibration_validated = True
                        latest = pose.latest()
                        calibration_finished_sequence = latest.sequence if latest else None
                    if now-last_status>.15:
                        skeleton_calibrated=dev.calibration_status(name=TargetName.SKELETON).state.value=='ready'
                        status={'type':'status','capture':None,'calibration':None,'pid':os.getpid(),'sdk_creations':sdk_creations}
                        status['calibration_review_scale'] = review_scale
                        if capture_session:
                            snap=capture_session.snapshot()
                            stage=next((s for s in snap.stage_progress if s.stage_id==snap.current_stage),None)
                            if stage:
                                prompt=next((s.prompt for s in snap.plan.stages if s.stage_id==stage.stage_id),stage.prompt)
                                status['capture']={'session_state':snap.state.value,'failure':snap.failure.message if snap.failure else None,'stage_id':stage.stage_id,'prompt':prompt,'state':stage.state.value,'collected':stage.collected_sample_count,'required':stage.required_sample_count,'request_id':snap.request_id}
                        elif current:
                            c=current.value;s=c.stage
                            status['capture']={'session_state':c.session_state.value,'failure':c.failure.message if c.failure else None,'stage_id':s.stage_id,'prompt':s.prompt,'state':s.state.value,'collected':s.collected_sample_count,'required':s.required_sample_count,'request_id':s.request_id}
                        job=calibration.latest()
                        if calibration_session:
                            snap=calibration_session.snapshot();failed=next((j.failure for j in snap.jobs if j.failure),None)
                            status['calibration']={'state':snap.state.value,'failure':failed.message if failed else None}
                        elif job:
                            j=job.value.job;status['calibration']={'state':j.state.value,'failure':j.failure.message if j.failure else None}
                        send(status);last_status=now
                    r=raw.latest();s=pose.latest();a=angles.latest();job=calibration.latest()
                    if s and s.sequence!=last_skeleton_sequence:
                        last_skeleton_sequence=s.sequence;last_skeleton_update=now
                    if a and a.sequence!=last_angle_sequence:
                        last_angle_sequence=a.sequence;last_angle_update=now
                    if capture_session:
                        capturing=capture_session.snapshot().state.value in ('running','failed','cancelled')
                        calculating=calibration_session is None or calibration_session.snapshot().state.value!='completed'
                    else:
                        capturing=current and current.value.session_state.value=='running'
                        calculating=job and job.value.job.state.value in ('pending','running','failed')
                    # Discard SDK Skeleton samples cached during the official capture/calibration.
                    skeleton_ready = bool(
                        not capturing and not calculating and skeleton_calibrated and s is not None
                        and now - last_skeleton_update < .5
                        and s.sequence != calibration_finished_sequence
                        and (capture_session is None or calibration_validated)
                    )
                    preview = not skeleton_ready
                    angles_fresh = skeleton_ready and a is not None and now-last_angle_update<.5
                    source='qnbot_raw_calibration_preview' if preview else 'qnbot_calibrated_skeleton'
                    reviewing_frame = skeleton_ready and review_scale is not None
                    key=(source,r.sequence if preview and r else s.sequence if s else -1,last_angle_sequence,angles_fresh,last_skeleton_sequence,skeleton_ready,reviewing_frame)
                    if r and (preview or s) and key!=last_key and now-last_send>.015:
                        payload=r.value.payload
                        nodes=dict(zip(s.value.names,s.value.positions_local)) if skeleton_ready else None
                        if preview:positions={k.value:v.position for k,v in payload.fingertip_local.items()}
                        else:
                            nodes=dict(zip(s.value.names,s.value.positions_local));wrist=nodes['wrist']
                            names={'thumb':'thumb_tip','index':'index_tip','middle':'middle_tip','ring':'ring_tip','pinky':'little_tip'}
                            positions={k:[nodes[n][i]-wrist[i] for i in range(3)] for k,n in names.items()}
                        frame_id+=1
                        send({'type':'frame','sequence':frame_id,'sdk_sequence':key[1],'raw_sequence':r.sequence,'joint_sequence':last_angle_sequence,'joints_fresh':angles_fresh,'skeleton_sequence':last_skeleton_sequence,'skeleton_ready':skeleton_ready,'calibration_review':reviewing_frame,'valid':{k.value:v for k,v in payload.raw_valid_mask.items()},'held':{k.value:v for k,v in payload.held_mask.items()},'positions':positions,'source':source,'skeleton_nodes':nodes if skeleton_ready else None,'joints':a.value.joints if angles_fresh else None})
                        last_key=key;last_send=now
                    time.sleep(.003)
            finally:
                # Also runs if Skeleton setup/start/manual calibration throws.
                close_sessions()
                if glove is not None:
                    try:glove.close()
                    except Exception as exc:print(f'glove.close failed: {exc}',file=sys.stderr,flush=True)
                if sdk is not None:
                    try:sdk.close()
                    except Exception as exc:print(f'sdk.close failed: {exc}',file=sys.stderr,flush=True)
                print(f'SDK CLOSED pid={os.getpid()} sdk_creations={sdk_creations}',flush=True)
                guard.rollback()
    except CalibrationRejected as exc:
        message = f'{exc}。已恢复重采前的数据，请关闭窗口后重新启动仿真。'
        try:
            send({'type': 'error', 'message': message, 'fatal': True})
        except OSError:
            pass
        print(message, file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        message=f'{args.port}: {exc}。连接已停止；不会自动重试。请检查已有窗口、VSCode 调试与 USB 连接。'
        try:send({'type':'error','message':message,'fatal':True})
        except OSError:pass
        print(message,file=sys.stderr,flush=True)
        return 1
    finally:sock.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
