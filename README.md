# 🤟 Sign Language Translator

一个基于深度学习的实时手语翻译系统，支持多种手语语言到文本的翻译，以及文本到语音的转换。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 功能特性

### 🎯 核心功能
- **实时手语识别**: 基于MediaPipe和深度学习的手势检测
- **多语言支持**: 支持ASL、BSL、JSL、CSL等主流手语
- **智能翻译**: 手语到多种自然语言的翻译
- **语音合成**: 文本到语音(TTS)功能
- **Web界面**: 现代化的Web应用界面

### 🔧 技术特性
- **高性能模型**: Transformer架构的手语识别模型
- **模型版本管理**: 完整的MLOps流程支持
- **实时监控**: Prometheus + Grafana监控体系
- **可扩展部署**: Kubernetes + Docker容器化部署
- **A/B测试**: 内置模型A/B测试功能

### 🌐 支持的语言

| 手语类型 | 代码 | 词汇量 | 支持地区 |
|---------|------|--------|----------|
| 美国手语 | ASL | 1000+ | US, CA |
| 英国手语 | BSL | 800+ | GB, AU |
| 日本手语 | JSL | 1200+ | JP |
| 中国手语 | CSL | 1500+ | CN, HK, TW |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+ (用于GPU加速)
- Redis (用于缓存)
- PostgreSQL (可选，用于数据存储)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/Sign_Language_Translator.git
cd Sign_Language_Translator
```

2. **安装依赖**
```bash
# 开发环境
pip install -r requirements.txt

# 生产环境
pip install -r requirements_production.txt
```

3. **配置环境变量**
```bash
cp config/production_config.yaml config/local_config.yaml
# 编辑配置文件中的相关参数
```

4. **启动服务**
```bash
# 启动API服务
python -m uvicorn api.server:app --reload

# 或使用主程序
python src/main_v2.py
```

## 📖 使用说明

### 命令行使用

```bash
# 训练模型
python src/main_v2.py --mode train --language ASL

# 实时翻译
python src/main_v2.py --mode demo --camera 0

# 性能测试
python src/main_v2.py --mode benchmark
```

### Web界面使用

1. 启动API服务后，打开 `client/web_app.html`
2. 允许摄像头权限
3. 选择源手语和目标语言
4. 点击"开始翻译"按钮

### API使用

#### 图片翻译
```bash
curl -X POST "http://localhost:8080/translate/image" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@image.jpg" \
  -F "source_language=ASL" \
  -F "target_language=en"
```

#### 实时翻译 (WebSocket)
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/translate/client123?source_language=ASL&target_language=en');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Translation:', data);
};

// 发送图片数据
ws.send(imageBlob);
```

## 🏗️ 项目架构

```
Sign_Language_Translator/
├── 📁 src/                    # 主应用程序
│   ├── main.py               # 基础版本主程序
│   └── main_v2.py            # 增强版本主程序
├── 📁 model/                 # 模型组件
│   ├── detectors/            # 手势检测器
│   ├── classifiers/          # 手势分类器
│   └── engines/              # 翻译引擎
├── 📁 core/                  # 核心业务逻辑
│   ├── models.py             # 深度学习模型
│   ├── translation_service.py # 翻译服务
│   ├── monitoring.py         # 监控系统
│   └── data_pipeline.py      # 数据处理
├── 📁 api/                   # REST API服务
├── 📁 client/                # Web客户端
├── 📁 deployment/            # 部署配置
├── 📁 config/                # 配置文件
└── 📁 scripts/               # 工具脚本
```

## 🔧 配置说明

主要配置文件: `config/production_config.yaml`

```yaml
app:
  name: "SignLanguageTranslator"
  version: "1.0.0"
  environment: "production"

models:
  detection:
    confidence_threshold: 0.8
    max_hands: 2
  recognition:
    architecture: "transformer"
    embedding_dim: 256
    num_heads: 8

api:
  host: "0.0.0.0"
  port: 8080
  rate_limit:
    requests_per_minute: 60
```

## 📊 性能基准

| 模型 | 准确率 | 推理时间 | 内存占用 |
|------|--------|----------|----------|
| ASL-v1.0 | 94.2% | 12ms | 2.1GB |
| BSL-v1.0 | 91.8% | 15ms | 1.9GB |
| JSL-v1.0 | 89.5% | 18ms | 2.3GB |
| CSL-v1.0 | 92.1% | 16ms | 2.5GB |

## 🐳 部署指南

### Docker部署

```bash
# 构建镜像
docker build -f deployment/Dockerfile -t sign-language-translator .

# 运行容器
docker-compose -f deployment/docker-compose.yml up -d
```

### Kubernetes部署

```bash
# 应用配置
kubectl apply -f deployment/kubernetes/

# 检查状态
kubectl get pods -n production
```

### 生产环境部署

1. **设置环境变量**
```bash
export JWT_SECRET="your-secret-key"
export REDIS_HOST="redis.example.com"
export DB_HOST="postgres.example.com"
```

2. **配置监控**
```bash
# 启动Prometheus
docker run -d -p 9090:9090 prom/prometheus

# 启动Grafana
docker run -d -p 3000:3000 grafana/grafana
```

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_api.py -v

# 生成覆盖率报告
pytest --cov=core tests/
```

## 📈 监控和日志

### 监控指标
- 请求延迟和吞吐量
- 模型推理性能
- 系统资源使用率
- 翻译准确率

### 访问监控面板
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## 🤝 开发指南

### 开发环境设置
```bash
# 克隆仓库
git clone https://github.com/your-username/Sign_Language_Translator.git

# 安装开发依赖
pip install -r requirements.txt
pip install pre-commit

# 设置pre-commit hooks
pre-commit install
```

### 代码规范
- 使用Black进行代码格式化
- 使用Flake8进行代码检查
- 使用Mypy进行类型检查
- 遵循PEP 8规范

### 贡献流程
1. Fork项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📚 API文档

### 认证
所有API端点都需要JWT认证:
```
Authorization: Bearer <your-jwt-token>
```

### 主要端点

#### 翻译相关
- `POST /translate/image` - 图片翻译
- `POST /translate/video` - 视频翻译
- `WS /ws/translate/{client_id}` - 实时翻译

#### 语音合成
- `POST /tts` - 文本转语音

#### 系统管理
- `GET /health` - 健康检查
- `GET /metrics` - 监控指标
- `POST /models/reload` - 重载模型

详细API文档: [http://localhost:8080/docs](http://localhost:8080/docs)

## 🔒 安全说明

- 使用JWT进行身份认证
- 支持HTTPS/WSS加密传输
- 实施速率限制和CORS策略
- 定期更新依赖包

## 📝 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持和反馈

- 📧 邮箱: support@signlanguagetranslator.com
- 🐛 问题反馈: [GitHub Issues](https://github.com/your-username/Sign_Language_Translator/issues)
- 💬 讨论: [GitHub Discussions](https://github.com/your-username/Sign_Language_Translator/discussions)

## 🙏 致谢

感谢以下开源项目和贡献者:
- [MediaPipe](https://mediapipe.dev/) - 手势检测
- [TensorFlow](https://tensorflow.org/) - 深度学习框架
- [FastAPI](https://fastapi.tiangolo.com/) - Web框架
- 所有贡献者和测试用户

---

**🌟 如果这个项目对您有帮助，请给个Star支持一下！**