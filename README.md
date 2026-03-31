# TransNetV2 RabbitMQ Worker

基于 [TransNet V2](https://arxiv.org/abs/2008.04838) 的视频镜头边界检测服务，封装成一个消费 RabbitMQ 队列任务的 worker。任务视频从 S3 下载，处理结果和抽帧图片再上传回 S3。

## 功能

- 消费 RabbitMQ 队列中的视频处理任务
- 从 S3 兼容存储下载视频
- 使用 PyTorch 版 TransNetV2 做镜头切分
- 将结果 JSON、预览帧、音频文件以及可提取字幕上传到 S3
- 提供结果可视化脚本

## 项目结构

```text
.
|-- app/
|   |-- config.py
|   |-- predictor.py
|   |-- s3_client.py
|   `-- worker.py
|-- inference_pytorch/
|   `-- transnetv2_pytorch.py
|-- scripts/
|   `-- plot_result.py
|-- weights/
|-- docker-compose.yml
|-- Dockerfile
|-- main.py
`-- README.md
```

## 环境变量

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `RABBITMQ_URL` | RabbitMQ 连接地址 | `amqp://guest:guest@localhost:5672/` |
| `QUEUE_NAME` | 消费队列名 | `transnet_tasks` |
| `DONE_QUEUE_NAME` | 结果完成队列名 | `transnet_tasks_done` |
| `S3_ENDPOINT_URL` | S3 兼容端点 | 空 |
| `S3_ACCESS_KEY` | S3 Access Key | 空 |
| `S3_SECRET_KEY` | S3 Secret Key | 空 |
| `S3_BUCKET` | Bucket 名称 | 空 |
| `S3_REGION` | 区域 | `us-east-1` |
| `USE_GPU` | 是否启用 GPU | `false` |
| `CUDA_VISIBLE_DEVICES` | GPU 设备 ID | 空 |
| `WEIGHTS_PATH` | 权重文件路径 | `./weights/transnetv2-pytorch-weights.pth` |
| `RESULT_PREFIX` | 结果前缀 | `results/` |
| `FRAME_IMAGE_PREFIX` | 抽帧图片前缀 | `frames/` |
| `AUDIO_PREFIX` | 音频文件前缀 | `audio/` |
| `SUBTITLE_PREFIX` | 字幕文件前缀 | `subtitles/` |
| `TEMP_DIR` | 本地临时目录 | `./.tmp` |

## 本地运行

1. 准备权重文件，放到 `weights/transnetv2-pytorch-weights.pth`
2. 复制环境变量模板

```bash
cp .env.example .env
```

3. 安装依赖

```bash
uv sync
```

4. 启动 worker

```bash
uv run python main.py
```

## Docker Compose 部署

仓库现在自带完整编排，包含这些服务：

- `worker`: TransNetV2 消费者
- `rabbitmq`: 消息队列，管理后台默认暴露在 `15672`
- `minio`: S3 兼容对象存储，控制台默认暴露在 `9001`
- `minio-init`: 自动创建 `S3_BUCKET`

### 1. 准备配置

复制模板并按实际环境修改：

```bash
cp .env.example .env
```

如果你直接使用仓库里的 `docker-compose.yml`，`.env` 建议至少确认这些值：

```env
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=videos
QUEUE_NAME=transnet_tasks
DONE_QUEUE_NAME=transnet_tasks_done
```

`docker-compose.yml` 会自动把容器内地址改成：

- RabbitMQ: `amqp://guest:guest@rabbitmq:5672/`
- MinIO: `http://minio:9000`

所以 `.env` 中的 `localhost` 配置不会影响容器间通信。

### 2. 准备模型权重

把模型权重放到：

```text
weights/transnetv2-pytorch-weights.pth
```

Compose 会把这个目录只读挂载到容器内的 `/app/weights`。

### 3. 启动

```bash
docker compose up -d --build
```

### 4. 查看状态

```bash
docker compose ps
docker compose logs -f worker
```

### 5. 访问配套服务

- RabbitMQ 管理页: `http://localhost:15672`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`

### 6. 停止

```bash
docker compose down
```

如果需要连同持久化数据一起清理：

```bash
docker compose down -v
```

## 单容器运行

如果 RabbitMQ 和 S3 已经由外部环境提供，也可以只启动 worker：

```bash
docker build -t transnetv2-worker .
docker run -d \
  --env-file .env \
  -v $(pwd)/weights:/app/weights:ro \
  transnetv2-worker
```

这种方式下，需要确保 `.env` 中的 `RABBITMQ_URL` 和 `S3_ENDPOINT_URL` 指向真实可达的外部服务，而不是容器内默认地址。

## 消息格式

输入消息：

```json
{
  "task_id": "optional-task-uuid",
  "s3_key": "videos/example.mp4",
  "scene_threshold": 0.5,
  "max_scene_sample_interval_seconds": 5.0,
  "extract_audio": true,
  "extract_subtitles": true
}
```

字段说明：

- `task_id`: 可选，不传时自动生成
- `s3_key`: 必填，待处理视频在 S3 中的 key
- `scene_threshold`: 可选，镜头切分阈值，默认 `0.5`
- `max_scene_sample_interval_seconds`: 可选，同一镜头内最大抽样间隔秒数，默认 `5.0`
- `extract_audio`: 可选，是否提取第一个音频流并上传到 S3，默认 `true`
- `extract_subtitles`: 可选，是否探测并提取可转换的内封字幕流，默认 `true`

输出结果会上传到：

```text
results/{task_id}/result.json
```

任务处理成功后，还会将与 `result.json` 内容完全一致的 JSON 消息发送到 RabbitMQ 结果队列，默认值为：

```text
transnet_tasks_done
```

示例：

```json
{
  "task_id": "xxx",
  "s3_key": "videos/example.mp4",
  "frame_count": 1500,
  "fps": 25.0,
  "source_container": "mov,mp4,m4a,3gp,3g2,mj2",
  "source_extension": ".mp4",
  "scene_threshold": 0.5,
  "max_scene_sample_interval_seconds": 5.0,
  "audio": {
    "stream_index": 1,
    "codec_name": "aac",
    "channels": 2,
    "sample_rate": 48000,
    "bit_rate": 128000,
    "language": "und",
    "audio_key": "audio/xxx/audio.m4a"
  },
  "subtitles": [
    {
      "stream_index": 2,
      "codec_name": "mov_text",
      "language": "chi",
      "title": "Simplified Chinese",
      "default": true,
      "forced": false,
      "extractable": true,
      "subtitle_key": "subtitles/xxx/subtitle-2.srt"
    }
  ],
  "scene_preview_frames": [
    {
      "scene_index": 0,
      "start_frame": 0,
      "end_frame": 449,
      "sample_count": 2,
      "sampled_frames": [
        {
          "sample_index": 0,
          "frame_id": 112,
          "image_key": "s3://videos/frames/xxx/112.png"
        }
      ]
    }
  ],
  "result_key": "results/xxx/result.json"
}
```

说明：

- 视频推理仍然直接把视频流解码成模型所需低分辨率帧，不会先把整段视频导出成完整帧序列落盘。
- 音频提取优先使用流拷贝，不重新编码，尽量减少 CPU 和耗时。
- 字幕会先探测内封字幕流；当前默认提取文本类字幕编码，如 `mov_text`、`subrip`、`ass`、`webvtt`。图片类字幕流会保留元数据，但不会强制转码提取。
- 输入不只支持 `mp4`。只要当前环境里的 `ffmpeg` 能解封装和解码，`mov`、`mkv`、`avi`、`webm` 等都可以处理；最终支持范围取决于运行时 `ffmpeg` 编译能力。

## 结果可视化

```bash
uv run python scripts/plot_result.py \
  --input ~/Downloads/result.json \
  --output ~/Downloads/result_plot.png
```

可选参数：

- `--threshold 0.5`
- `--zoom-start 300 --zoom-end 500`

## ACK 机制

- 处理成功: `basic_ack`
- 消息格式错误: `basic_nack(requeue=False)`
- 处理异常: `basic_nack(requeue=True)`

## 引用

```bibtex
@article{soucek2020transnetv2,
    title={TransNet V2: An effective deep network architecture for fast shot transition detection},
    author={Sou{\v{c}}ek, Tom{\'a}{\v{s}} and Loko{\v{c}}, Jakub},
    year={2020},
    journal={arXiv preprint arXiv:2008.04838},
}
```
