# TransNetV2 RabbitMQ Worker

基于 [TransNet V2](https://arxiv.org/abs/2008.04838) 的视频镜头边界检测服务，封装为 RabbitMQ 消费者。

## 功能

- 消费 RabbitMQ 队列中的视频处理任务
- 从 S3 下载视频文件
- 使用 PyTorch 版 TransNetV2 进行镜头边界检测
- 将结果上传到 S3
- 提供本地结果分析脚本

## 项目结构

```text
.
├── app/
│   ├── __init__.py
│   ├── config.py              # 环境变量配置
│   ├── predictor.py           # 推理封装
│   ├── s3_client.py           # S3 操作
│   └── worker.py              # RabbitMQ 消费者
├── inference_pytorch/
│   ├── __init__.py
│   └── transnetv2_pytorch.py  # TransNetV2 模型
├── scripts/
│   └── plot_result.py         # 结果可视化脚本
├── weights/
│   └── transnetv2-pytorch-weights.pth
├── main.py
├── start_worker.sh
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── .env.example
└── README.md
```

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `RABBITMQ_URL` | RabbitMQ 连接 URL | `amqp://guest:guest@localhost:5672/` |
| `QUEUE_NAME` | 消费队列名称 | `transnet_tasks` |
| `S3_ENDPOINT_URL` | S3 端点 (MinIO/其他) | `` |
| `S3_ACCESS_KEY` | S3 Access Key | - |
| `S3_SECRET_KEY` | S3 Secret Key | - |
| `S3_BUCKET` | S3 Bucket 名称 | - |
| `S3_REGION` | S3 区域 | `us-east-1` |
| `USE_GPU` | 是否使用 GPU | `false` |
| `CUDA_VISIBLE_DEVICES` | GPU 设备 ID | - |
| `WEIGHTS_PATH` | 模型权重路径 | `./weights/transnetv2-pytorch-weights.pth` |
| `RESULT_PREFIX` | 结果文件前缀 | `results/` |
| `FRAME_IMAGE_PREFIX` | 帧图片前缀 | `frames/` |

## 快速开始

### 1. 准备模型权重

将权重文件放到 `weights/transnetv2-pytorch-weights.pth`。

### 2. 配置环境变量

```bash
cp .env.example .env
```

按实际环境填写 `.env` 中的 RabbitMQ、S3 和权重路径配置。

### 3. 安装依赖

```bash
uv sync
```

### 4. 启动 worker

```bash
./start_worker.sh
```

如果不使用启动脚本，也可以直接运行：

```bash
uv run python main.py
```

### 5. Docker 运行

```bash
docker build -t transnetv2-worker .
docker run -d \
  --env-file .env \
  -v $(pwd)/weights:/app/weights \
  transnetv2-worker
```

## 消息格式

### 输入消息

```json
{
  "task_id": "optional-task-uuid",
  "s3_key": "videos/example.mp4",
  "scene_threshold": 0.5,
  "max_scene_sample_interval_seconds": 5.0
}
```

说明：

- `task_id` 可选，不传时由 worker 自动生成
- `s3_key` 必填，指向待处理视频
- `scene_threshold` 可选，镜头切分阈值，默认 `0.5`
- `max_scene_sample_interval_seconds` 可选，同一镜头内抽图的最大时间间隔，默认 `5.0`

### 输出结果

worker 成功处理后会将结果上传到：

`results/{task_id}/result.json`

```json
{
  "task_id": "xxx",
  "s3_key": "videos/example.mp4",
  "frame_count": 1500,
  "scenes": [[0, 100], [101, 250]],
  "single_frame_predictions": [0.1, 0.2, 0.9],
  "all_frame_predictions": [0.1, 0.2, 0.8],
  "scene_threshold": 0.5,
  "max_scene_sample_interval_seconds": 5.0,
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
          "image_key": "frames/xxx/112.png"
        },
        {
          "sample_index": 1,
          "frame_id": 337,
          "image_key": "frames/xxx/337.png"
        }
      ]
    }
  ]
}
```

说明：

- `scenes` 是按帧号表示的镜头区间
- `single_frame_predictions` 是逐帧切点分数
- `all_frame_predictions` 是逐帧转场区域分数
- `scene_threshold` 是本次任务实际使用的切分阈值
- `max_scene_sample_interval_seconds` 是本次任务实际使用的镜头抽图最大间隔
- `scene_preview_frames` 是每个镜头对应的抽图结果
- `frame_id` 会直接作为上传图片文件名的一部分，例如 `frames/{task_id}/{frame_id}.png`
- `image_key` 是上传到 S3 的对象 key

`TransNetWorker.run_once()` 的返回值还会包含：

```json
{
  "result_key": "results/{task_id}/result.json"
}
```

当前主消费链路默认只上传 `result.json`，不再上传可视化 PNG。

## 镜头抽图规则

worker 会为每个分析出的镜头自动抽取代表帧并上传到 S3。

规则如下：

- 单个镜头内相邻采样点的最大时间间隔不超过 `max_scene_sample_interval_seconds`
- 采样数量按 `ceil(镜头时长 / 最大间隔)` 计算
- 然后把整个镜头平均分成这么多段
- 每一段取中间帧作为代表图
- 需要上传的帧会先批量提取，再并发上传到 S3

例如：

- 镜头时长 `4s`，默认参数下抽 `1` 张
- 镜头时长 `10s`，默认参数下抽 `2` 张
- 镜头时长 `15s`，默认参数下抽 `3` 张

## 结果分析

项目提供了 [scripts/plot_result.py](./scripts/plot_result.py) 用于分析 `result.json`。

示例：

```bash
uv run python scripts/plot_result.py \
  --input ~/Downloads/result.json \
  --output ~/Downloads/result_plot.png
```

可选参数：

- `--threshold 0.5` 绘制参考阈值线
- `--zoom-start 300 --zoom-end 500` 输出局部放大图

输出内容包括：

- `single_frame_predictions` 折线
- `all_frame_predictions` 折线
- scene 边界标记
- scene 区间可视化

## ACK 机制

- 处理成功: `basic_ack`
- 消息格式错误: `basic_nack(requeue=False)`
- 处理异常: `basic_nack(requeue=True)` 重新入队

## 引用

```bibtex
@article{soucek2020transnetv2,
    title={TransNet V2: An effective deep network architecture for fast shot transition detection},
    author={Sou{\v{c}}ek, Tom{\'a}{\v{s}} and Loko{\v{c}}, Jakub},
    year={2020},
    journal={arXiv preprint arXiv:2008.04838},
}
```
