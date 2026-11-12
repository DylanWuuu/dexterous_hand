"""Windows project ownership guards. Locks disappear when their process exits."""
import ctypes,json,os
from ctypes import wintypes
from pathlib import Path

k32=ctypes.WinDLL('kernel32',use_last_error=True)
k32.CreateMutexW.argtypes=[ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR];k32.CreateMutexW.restype=wintypes.HANDLE
k32.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k32.WaitForSingleObject.restype=wintypes.DWORD
k32.CloseHandle.argtypes=[wintypes.HANDLE]
k32.ReleaseMutex.argtypes=[wintypes.HANDLE]
k32.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k32.OpenProcess.restype=wintypes.HANDLE
k32.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];k32.CreateFileW.restype=wintypes.HANDLE

class InstanceLock:
    def __init__(self,role,port):
        port=port.upper();self.path=Path(__file__).resolve().parent/'generated'/f'{role}_{port}_owner.json';self.handle=None
        h=k32.CreateMutexW(None,False,'Local\\QnbotDexterous_'+role+'_'+port)
        if not h:raise ctypes.WinError(ctypes.get_last_error())
        wait=k32.WaitForSingleObject(h,0)
        if wait not in (0,0x80):
            k32.CloseHandle(h)
            try:owner=json.loads(self.path.read_text(encoding='utf-8')).get('pid','未知')
            except (OSError,ValueError):owner='未知'
            raise RuntimeError(f'{port} 已有本项目 {role} 实例（PID {owner}）。请关闭已有窗口/VSCode Debug，再重新启动；本次不会打开串口。')
        self.handle=h
        try:
            self.path.parent.mkdir(exist_ok=True);self.path.write_text(json.dumps({'pid':os.getpid(),'role':role,'port':port}),encoding='utf-8')
        except BaseException:self.close();raise
    def close(self):
        if self.handle:
            try:self.path.unlink(missing_ok=True)
            finally:k32.ReleaseMutex(self.handle);k32.CloseHandle(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

class ParentWatch:
    def __init__(self,pid):
        self.handle=k32.OpenProcess(0x00100000,False,pid) if pid else None
        if pid and not self.handle:raise RuntimeError(f'无法监视主程序 PID {pid}，不打开串口')
    def alive(self):return not self.handle or k32.WaitForSingleObject(self.handle,0)==0x102
    def close(self):
        if self.handle:k32.CloseHandle(self.handle);self.handle=None

def detect_glove_port(side):
    from qnbot_sdk.glove import discover_gloves

    gloves = discover_gloves()
    ports = [glove.port for glove in gloves if glove.error is None
             and (glove.hand is None or glove.hand.value == side)]
    if len(ports) == 1:
        return ports[0]
    if ports:
        raise RuntimeError(f'检测到多个匹配的手套（{", ".join(ports)}），请用 --port 指定串口。')
    raise RuntimeError(f'未检测到可用的 {side} 手手套。请检查 USB 连接和串口占用，也可用 --port 指定串口。')


def check_port_available(port):
    from serial.tools import list_ports
    available={p.device.upper():p.description for p in list_ports.comports()}
    if port.upper() not in available:raise RuntimeError(f'{port} 不存在。当前端口：{", ".join(available) or "无"}；请检查 USB 或 --port 参数。')
    h=k32.CreateFileW('\\\\.\\'+port.upper(),0xC0000000,0,None,3,0,None)
    if h==ctypes.c_void_p(-1).value:
        error=ctypes.get_last_error()
        raise RuntimeError(f'{port} 无法独占打开（Windows {error}: {ctypes.FormatError(error).strip()}）。可能被其他窗口/Debug/串口软件占用或设备拒绝访问；停止连接，不自动重复重试。')
    k32.CloseHandle(h)
