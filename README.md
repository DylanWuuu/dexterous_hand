# Qnbot 手套驱动 MuJoCo 灵巧手

Windows x64 上的左手仿真。Qnbot SDK 输出标定后的人手骨骼，Dex Vector 将骨骼转换为灵巧手关节角，MuJoCo 显示跟随效果。

## 安装与启动

安装 [Miniforge](https://github.com/conda-forge/miniforge)，在 Miniforge Prompt 中进入仓库目录，运行：

```bat
conda env create --prefix "%LOCALAPPDATA%\dexterous-hand" -f environment.yml
conda activate "%LOCALAPPDATA%\dexterous-hand"
start_glove_sim.bat
```

运行启动脚本前，接入左手 Qnbot VetraGlove Pro，并关闭占用手套串口的 Qnbot Studio 或其他调试程序。程序自动检测手套并生成 MuJoCo 模型。需要指定端口时运行 `start_glove_sim.bat --port COM4`，把 `COM4` 换成 Windows 设备管理器显示的端口。

环境路径请尽量短。若安装 Qnbot SDK 时提示 Windows 路径过长，换用较短的 `--prefix` 路径重新创建环境，例如 `D:\envs\dex-hand`。

## 首次使用

1. 首次连接会打开 Qnbot 官方八步采集。按窗口提示摆好动作，再点「姿势稳定，确认本步」，保持到采集结束。第一步五指自然并拢，第三步只弯四指的指根约 90°，第四步自然握拳。 [Qnbot 八步采集说明](https://docs.qnbot.com/zh-CN/docs/vetra-glove-pro/sdk/python/latest/capture)
2. 官方标定完成并收到新的骨骼帧后，机器人开始跟随。按窗口提示采集七组 Dex 参考动作，为本次标定拟合目标映射。
3. 试做张开、握拳、单指弯曲和捏合。确认后点「保留新标定和映射」。下次启动会直接复用，连接后即可跟随。

重新采集期间，机器人保持当前姿态，SDK 负责采集和标定。收到本次标定后的新骨骼时，Dex、滤波和防碰撞从模型零位重新开始。标定与映射存放在本机的 `qnbot_data/`。

## 仿真范围

当前 URDF 为左手。Dex Vector 优化 20 个手指关节，固定手腕，并使用指节方向、关节限位、低通滤波及简化自碰撞约束。MuJoCo 用于姿态跟随显示，物理接触力关闭。

遇到串口或跟随异常，可在控制窗口查看状态，或检查 `generated/qnbot_bridge.log`。算法依赖与许可见 [环境清单](environment.yml) 和 [第三方许可](THIRD_PARTY_NOTICES.md)。
