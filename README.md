# TransNetV2 RabbitMQ Worker

基于 [TransNet V2](https://arxiv.org/abs/2008.04838) 的视频镜头边界检测服务，封装为 RabbitMQ 消费者。

## 功能

- 消费 RabbitMQ 队列中的视频处理任务
- 从 S3 下载视频文件
- 使用 PyTorch 版 TransNetV2 进行镜头边界检测
- 生成可视化帧图
- 结果上传到 S3

## 项目结构

```
.
├── app/
│   ├── __init__.py
│   ├── config.py          # 环境变量配置
│   ├── predictor.py       # 推理封装
│   ├── s3_client.py       # S3 操作
│   └── worker.py          # RabbitMQ 消费者
├── inference_pytorch/
│   ├── __init__.py
│   └── transnetv2_pytorch.py  # TransNetV2 模型
├── weights/
│   └── transnetv2-pytorch-weights.pth  # 模型权重
├── main.py                # 入口文件
├── Dockerfile
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
| `WEIGHTS_PATH` | 模型权重路径 | `/app/weights/transnetv2-pytorch-weights.pth` |
| `RESULT_PREFIX` | 结果文件前缀 | `results/` |
| `FRAME_IMAGE_PREFIX` | 帧图片前缀 | `frames/` |

## 快速开始

### 1. 准备模型权重

```bash
# 下载预训练权重 (需要先安装 tensorflow)
python -c "
from inference_pytorch.transnetv2_pytorch import TransNetV2
import torch
# 使用原始仓库的转换脚本
"
```

将权重文件放到 `weights/transnetv2-pytorch-weights.pth`

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件
```

### 3. 本地运行

```bash
pip install -r requirements.txt
python main.py
```

### 4. Docker 运行

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
  "s3_key": "videos/example.mp4"
}
```

### 输出结果

上传到 S3 的结果文件:

**results/{task_id}/result.json**
```json
{
  "task_id": "xxx",
  "s3_key": "videos/example.mp4",
  "frame_count": 1500,
  "scenes": [[0, 100], [100, 250], ...],
  "single_frame_predictions": [0.1, 0.2, ...],
  "all_frame_predictions": [0.1, 0.2, ...],
  "result_key": "results/xxx/result.json",
  "visualization_key": "frames/xxx/visualization.png"
}
```

**frames/{task_id}/visualization.png** - 预测可视化图

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
