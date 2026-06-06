# LLM Proxy Inspector

OpenAI-compatible 反向代理 + 请求/响应可视化查看器。

## 安装与启动

使用 [uv](https://github.com/astral-sh/uv) 管理虚拟环境和依赖：

```bash
# 1. 创建虚拟环境（只执行一次）
uv venv

# 2. 安装依赖（uv 自动使用虚拟环境）
uv pip install fastapi uvicorn httpx

# 3. 运行脚本
uv run python proxy.py --upstream http://127.0.0.1:8008

# 其他参数示例
uv run python proxy.py --upstream http://127.0.0.1:8008 --proxy-port 8000 --ui-port 8001 --think off --params '{"temperature": 0.7}'
```

默认配置：上游 `http://127.0.0.1:8008`，代理端口 `:8000`，UI 端口 `:8001`。

## 使用

- 客户端将 API 地址指向 `http://<your-host>:8000`
- 浏览器打开 `http://<your-host>:8001` 查看请求/响应

## 截图

**消息双栏视图（Request / Response）**

![消息视图](docs/message.png)

**Raw JSON 视图**

![Raw JSON](docs/rawjson.png)

## 功能

- [x] 透传所有 HTTP 方法，原始数据不变
- [x] 流式 SSE 实时转发，结束后自动合并解析
- [x] 非流式 JSON 响应直接展示
- [x] 消息双栏视图（Request / Response）
- [x] Raw JSON 视图，支持一键复制
- [x] 思考链（reasoning）折叠展示
- [x] 工具调用（tool call）折叠展示
- [x] 侧边栏 5 秒局部刷新，不影响当前 tab
- [x] URL 格式 `/ids/<record_id>` 可分享

## License

[MIT](LICENSE)

## 目录结构

```
llm-proxy/
├── proxy.py          # 主程序（代理 + UI 服务）
├── requirements.txt
└── static/
    └── index.html    # 单文件前端
```

## OpenAI SDK 做法

关于 SSE stream 转 JSON, OpenAI Python SDK 不用通用的 _merge_delta，而是用强类型的 Pydantic 模型 + 专用 accumulate_delta 函数，按字段路径硬编码规则：

```
# openai/lib/_parsing/_completions.py 简化逻辑
# 只有这些路径会做拼接：
#   choice.delta.content
#   choice.delta.tool_calls[i].function.arguments
# 其余字段（type, id, role, name...）只在首次出现时设置，后续 chunk 不重复发送
```

关键在于：OpenAI 流式协议本身保证 type/id/role 这类字段只在首个 chunk 出现，后续 chunk 里就不会再有这些字段，所以官方 SDK 根本不用处理"重复覆盖"的问题。

而第三方 OpenAI-compatible 上游返回的 SSE 流可能在每个 chunk 里都带了 type: "function"，这本身是上游行为问题，但代理需要容错处理。
