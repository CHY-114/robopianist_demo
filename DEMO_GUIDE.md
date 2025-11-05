# 🎬 Piano-Drum Demo Guide

本指南介绍如何运行Piano-Drum环境的演示脚本，包括视觉展示、音频合成和视频录制。

## 📋 演示脚本概览

我们提供了三个不同复杂度的演示脚本：

| 脚本 | 复杂度 | 功能 | 推荐用途 |
|------|--------|------|---------|
| `demo_piano_drum_simple.py` | 低 | 基础动作展示 | 快速测试 |
| `demo_piano_drum_with_video.py` | 中 | 视频录制 | 创建演示视频 |
| `demo_piano_drum_full.py` | 高 | 完整音视频 | 完整展示 |

## 🚀 快速开始

### 1. 简单演示 (推荐入门)

```bash
cd /Users/simon/code/ML/CS5478/robopianist_demo

# 基础运行（无录制）
python examples/demo_piano_drum_simple.py

# 显示详细信息
python examples/demo_piano_drum_simple.py --show_details

# 捕获帧（保存为图片）
PYTHONPATH=$(pwd) python examples/demo_piano_drum_simple.py --capture_frames --output_dir ./frames

# 运行更多步数
python examples/demo_piano_drum_simple.py --n_steps 600
```

**输出示例：**
```
============================================================
🎹🥁 Piano-Drum Combined Environment Demo
============================================================

[1/4] Creating environment...
✓ Environment created successfully

Environment details:
  • Total action dimensions: 61
  • Observation dimensions: 324

Action space breakdown:
  • piano_right_hand    : 24 dimensions
  • piano_left_hand     : 24 dimensions
  • drum_right_stick    :  6 dimensions
  • drum_left_stick     :  6 dimensions
  • piano_sustain       :  1 dimensions

[4/4] Running simulation for 300 steps...
----------------------------------------------------------------------
Step   0/300: Piano= 3 keys, Drum=1 strikes, Reward=0.000
Step  30/300: Piano= 3 keys, Drum=1 strikes, Reward=0.000
...

📊 Simulation Statistics:
  • Total steps: 300
  • Piano activations: 75 steps (25.0%)
  • Drum activations: 120 steps (40.0%)
```

### 2. 视频录制演示 (mediapy版本)

需要先安装mediapy：
```bash
pip install mediapy
```

然后运行：
```bash
# 基础录制
python examples/demo_piano_drum_with_video.py --record

# 自定义输出目录
python examples/demo_piano_drum_with_video.py --record --output_dir ./my_videos

# 自定义分辨率和FPS
python examples/demo_piano_drum_with_video.py --record \
    --resolution 1920,1080 \
    --fps 30

# 运行更多步数（更长视频）
python examples/demo_piano_drum_with_video.py --record --n_steps 600
```

### 3. 完整音视频演示 (推荐)

这个版本包含完整的MIDI音频合成：

```bash
# 运行但不录制（快速测试）
python examples/demo_piano_drum_full.py

# 录制带音频的视频
python examples/demo_piano_drum_full.py --record

# 自定义设置
python examples/demo_piano_drum_full.py --record \
    --output_dir ./videos \
    --n_steps 600 \
    --fps 20 \
    --resolution 1280,720
```

**需要的依赖：**
- OpenCV或imageio（视频编码）
- ffmpeg（音视频合成）

安装方法：
```bash
# OpenCV
pip install opencv-python

# 或 imageio
pip install imageio

# ffmpeg (macOS)
brew install ffmpeg

# ffmpeg (Ubuntu/Debian)
sudo apt-get install ffmpeg
```

## 🎵 演示内容说明

### 音乐模式

所有演示脚本都实现了协调的piano-drum演奏：

#### Piano部分
- **和弦进行**：C → G → Am → F（流行音乐常用进行）
- **节奏**：每个和弦持续约1秒（20步 @ 20Hz）
- **音符**：使用基础三和音（根音、三度、五度）
- **Sustain**：选择性使用延音踏板

#### Drum部分
- **基础节奏**：Rock beat模式
- **Right Stick**：
  - Hi-hat：每拍都打
  - Snare：反拍（beat 2和4）
- **Left Stick**：
  - Kick：正拍（beat 1和3）
  - Crash：小节开始
  - Ride：装饰音

### 动作序列可视化

```
Time:  |----1----|----2----|----3----|----4----|
Piano: C-Major   G-Major   A-minor   F-Major
       ~~~~~~~~  ~~~~~~~~  ~~~~~~~~  ~~~~~~~~

Drum:  K..S..K.  ..K..S..  K..S..K.  ..K..S..
       HHHHHHHH  HHHHHHHH  HHHHHHHH  HHHHHHHH

K = Kick, S = Snare, H = Hi-hat
. = Rest, ~ = Piano held
```

## 📊 输出文件

### Simple Demo
```
./frames/
├── frame_0000.png
├── frame_0005.png
├── frame_0010.png
└── ...
```

### Video Demo (mediapy)
```
./piano_drum_videos/
└── piano_drum_demo.mp4    # 视频文件（无音频）
```

### Full Demo
```
./piano_drum_videos/
├── episode_000.mp4        # 视频+音频
├── episode_001.mp4
└── ...
```

## 🎛️ 命令行参数

### demo_piano_drum_simple.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--capture_frames` | bool | False | 捕获帧为图片 |
| `--output_dir` | str | ./piano_drum_frames | 输出目录 |
| `--n_steps` | int | 300 | 运行步数 |
| `--show_details` | bool | True | 显示详细信息 |

### demo_piano_drum_with_video.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--record` | bool | False | 录制视频 |
| `--output_dir` | str | ./piano_drum_videos | 输出目录 |
| `--n_steps` | int | 400 | 运行步数 |
| `--fps` | int | 20 | 视频FPS |
| `--resolution` | list | [1280, 720] | 视频分辨率 |

### demo_piano_drum_full.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--record` | bool | False | 录制视频+音频 |
| `--output_dir` | str | ./piano_drum_videos | 输出目录 |
| `--n_steps` | int | 400 | 运行步数 |
| `--fps` | int | 20 | 视频FPS |
| `--resolution` | list | [1280, 720] | 视频分辨率 |

## 🔧 自定义演示

### 修改音乐模式

编辑demo脚本中的policy类：

```python
# 在 demo_piano_drum_simple.py 中
def create_piano_action(step, n_actuators):
    """修改这个函数来改变piano模式"""
    # 你的自定义代码
    pass

def create_drum_action(step):
    """修改这个函数来改变drum模式"""
    # 你的自定义代码
    pass
```

### 添加新的drum pattern

```python
# 示例：添加爵士鼓模式
def create_jazz_pattern(step):
    beat = step % 16

    # Swing rhythm on hi-hat
    if beat % 3 == 0:
        return hi_hat_strike
    elif beat % 3 == 2:
        return ride_strike
    # ... 更多pattern
```

### 修改视频设置

```python
# 在脚本中修改这些值
recorder = PianoDrumVideoRecorder(
    env=env,
    output_dir="./my_custom_videos",
    fps=30,              # 更高FPS
    resolution=(1920, 1080),  # 更高分辨率
)
```

## 🎥 视频质量优化

### 提高视频质量

1. **增加分辨率**：
   ```bash
   --resolution 1920,1080
   ```

2. **提高FPS**：
   ```bash
   --fps 30
   ```

3. **使用更好的编码器**：
   编辑 `piano_drum_video.py` 中的ffmpeg参数：
   ```python
   "-c:v", "libx264",    # 使用H.264
   "-preset", "slow",    # 更慢但质量更好
   "-crf", "18",         # 更低CRF = 更高质量
   ```

### 减小文件大小

1. **降低分辨率**：
   ```bash
   --resolution 854,480
   ```

2. **降低FPS**：
   ```bash
   --fps 15
   ```

3. **减少步数**：
   ```bash
   --n_steps 200
   ```

## 🐛 常见问题

### Q: ImportError: No module named 'mediapy'
**A:** 安装mediapy：
```bash
pip install mediapy
```

### Q: ffmpeg not found
**A:** 安装ffmpeg：
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### Q: OpenCV import error
**A:** 安装OpenCV：
```bash
pip install opencv-python
```

### Q: 视频没有声音
**A:** 检查：
1. ffmpeg是否已安装
2. 是否有MIDI事件生成（检查终端输出）
3. 音频文件是否成功创建

### Q: 性能太慢
**A:** 尝试：
1. 降低分辨率：`--resolution 640,480`
2. 使用primitive collisions：在环境创建时设置
3. 减少步数：`--n_steps 200`

### Q: 视频卡顿
**A:** 增加FPS或减少动作复杂度

## 📈 性能基准

在标准配置下（1280x720, 20 FPS）：

| 配置 | 实时速度 | 内存使用 |
|------|----------|----------|
| 无录制 | ~1.0x | ~500MB |
| 视频录制 | ~0.5x | ~800MB |
| 视频+音频 | ~0.3x | ~1GB |

## 🎯 下一步

1. **修改音乐模式**：尝试不同的和弦进行和节奏pattern
2. **添加新乐器**：扩展drum kit组件
3. **实现RL训练**：使用这些demo作为baseline
4. **创建编排系统**：设计更复杂的音乐序列

## 📚 相关文档

- [QUICK_START.md](QUICK_START.md) - 快速入门
- [PIANO_DRUM_IMPLEMENTATION.md](PIANO_DRUM_IMPLEMENTATION.md) - 详细实现
- [ARCHITECTURE.md](ARCHITECTURE.md) - 系统架构

---

**祝你演奏愉快！🎹🥁**
