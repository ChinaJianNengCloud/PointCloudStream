# Agent 通信与动作返回协议

Agent 通信链将图像与任务文本交给外部服务处理，再把返回的相对动作转换为机器人目标序列。仓库提供服务发现、消息传输和动作消费代码，但没有包含实际的模型推理服务。网络连通、处理协议兼容和机器人动作可执行是三个独立条件。

## 实现组成

| 文件 | 关键对象或函数 |
| --- | --- |
| [network_client.py](../app/utils/networking/network_client.py) | `discover_server()`、同步示例 `send_message()` |
| [network_server.py](../app/utils/networking/network_server.py) | `ZeroconfServer`、处理函数注入、客户端线程 |
| [op_thread.py](../app/threads/op_thread.py) | `DataSendToServerThread`、`RobotTcpOpThread` |
| [agent_callbacks.py](../app/callbacks/agent_callbacks.py) | 请求构造、进度显示、响应分派 |
| [entry.py](../app/entry.py) | `view_predicted_poses()` 相对动作累积 |

## 服务发现

客户端通过 Zeroconf 浏览 `_agent._tcp.local.` 服务。发现回调取 `ServiceInfo.addresses[0]`，按 IPv4 转成地址，并保存端口。多次发现会覆盖之前保存的地址，没有服务选择或优先级策略。

`discover_server()` 等待配置的 `discovery_timeout` 后关闭 Zeroconf，返回地址元组或 `None`。主入口设置为 2 秒，独立客户端示例的默认值为 5 秒。GUI 扫描回调直接调用这个函数，因此等待发生在界面线程中。

服务端在监听后注册服务名、地址和端口。示例配置将 `host` 设为 `0.0.0.0`，同一值还被用于服务广播；监听所有网卡的地址与客户端能够连接的实际网卡地址并不是同一种用途。

## TCP 消息边界

请求和响应采用相同外层封装：

```text
+--------------------------+-----------------------------+
| 4 字节无符号大端长度 N   | N 字节 pickle 消息体        |
+--------------------------+-----------------------------+

payload = pickle.dumps(message)
header  = len(payload).to_bytes(4, byteorder='big')
```

pickle 直接承载 Python 字典和 NumPy 数组，无需先把图像转为 JSON 数字列表。它也把协议绑定到双方可反序列化的 Python 对象类型；代码没有指定固定 pickle 协议版本或业务 schema 版本。

GUI 发送线程先发送长度，再按 4096 字节分块发送消息体。接收响应时先读取长度，再循环读取消息体，最后 `pickle.loads()`。同步 `send_message()` 使用同样结构，但把头部与请求体交给一次 `sendall()`，主要用于独立示例。

### 完整读取的边界

TCP 提供字节流，单次 `recv(4)` 不保证返回完整的四字节头部。客户端和服务端当前都直接使用一次 `recv(4)` 解码长度，没有循环补齐头部。

消息体读取使用固定 `recv(4096)`，没有按剩余长度收紧每次读取，也没有跨消息保留多读的字节。因此该实现依赖简单的顺序请求响应行为，不适合直接扩展为流水线发送多条消息。

当前传输层还具有以下行为：

- 不限制接收长度，没有应用级读写超时和重试策略。
- 请求体发送使用 `send()`，没有单独处理返回零字节的情况。
- 响应体提前断开时退出循环，仍尝试反序列化已收数据。
- “Receiving Data 100%”在反序列化之前发出，不保证接收到完整有效对象。
- 异常被转成 `{"status": "error", "message": ...}`，进度同时发出 Error。

这些是当前分帧实现的边界，不属于外部模型计算逻辑。

## 请求字段

GUI 的处理请求为：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `command` | 字符串 | 固定为 `process_pcd` |
| `prompt` | 字符串 | 预设的目标和方向描述，加上用户输入 |
| `image` | NumPy 数组 | 图像数据 |

命令名保留了 `pcd`，但当前 GUI 请求不含点云字段。文本前缀定义了与特定目标部位有关的方向语义，这些自然语言方向没有在客户端转换为机器人基座坐标；其解释依赖外部处理服务。

当前图像来源也不等于实时相机观测。发送回调用 `hasattr(self, 'robot')` 判断分支，而主窗口构造时已经创建 `robot` 属性，即使值为 `None`，仍会选择硬编码文件路径读取图片。另一个分支使用旧的 `current_frame['color'].cpu()`，与当前按相机名称组织的 NumPy 图像字典不同。

因此，请求构造代码保留了历史图像接口，不能将现有 Send 操作直接描述为发送当前多相机画面。

## 服务端处理函数契约

`ZeroconfServer.set_processing_function(func)` 接收一个处理回调。`handle_client()` 读取完整消息体后，把原始字节交给回调，回调需要返回已经序列化的响应字节：

```text
request_bytes
  → process_function(request_bytes)
  → response_bytes
  → 服务端添加四字节长度并发送
```

服务端不会替回调统一执行 `pickle.loads()` 或 `pickle.dumps()`。默认处理器返回序列化的错误消息，示例处理器则自行反序列化并处理命令。

每个连接分配一个 Python 线程，线程内可以顺序处理多次请求。每次收到请求后还固定等待 3 秒，属于示例行为，不是模型耗时统计。没有并发数限制；外部注入的处理函数会被不同客户端线程并发调用。

示例服务端处理 `process_pcd` 时要求 `pcd` 是 NumPy 数组，并返回 `status='success'` 和 `result`。GUI 则发送 `image` 并期待 `action`、`no_action` 或 `error`。两者共享传输封装，但业务字段与状态枚举并不匹配。

## 动作响应与累积语义

GUI 使用以下状态分支：

| `status` | 行为 |
| --- | --- |
| `action` | 将 `message` 解释为相对动作序列，转换后启动 TCP 动作线程 |
| `no_action` | 显示通用失败提示 |
| `error` | 显示通用失败提示 |

`view_predicted_poses()` 隐含要求每条动作至少包含七项：

```text
[dx, dy, dz, drx, dry, drz, stop]
```

前三项按米解释，后三个旋转分量按 `xyz` 欧拉角弧度解释。`stop == 1` 时立即终止循环，该条记录自身不执行。代码没有对形状、有限数值、幅度或坐标语义进行协议校验。

初始参考为响应到达后重新读取的机器人 TCP 状态，而不是发送请求时的状态。若请求处理期间机器人发生移动，相同相对动作会作用到新的起点。

数学上，函数意图使用 `align` 规则逐步累积：

```text
p_(i+1) = p_i + delta_p_i
R_(i+1) = delta_R_i · R_i
```

具体含义见[位姿表示与轨迹插值](pose-and-interpolation.md)。

### 当前动作链的接口限制

函数第一次迭代后把 `previous_pose` 赋成 NumPy 数组，第二次继续调用它的 `apply_delta_pose()` 会失败。因此，多条普通动作不能按现有实现完整累积。

响应回调把执行线程保存到局部变量 `thread` 后立即返回，没有像标定或关节回放那样在主窗口对象中保留引用。线程生命周期与窗口关闭流程也没有统一协调。

界面创建了 Reset 按钮，文件中也存在发送 `command='reset'` 的回调，但主窗口绑定逻辑没有连接该按钮；示例服务端同样没有 reset 处理分支。

## 信任边界与启动副作用

当前客户端直接反序列化响应，服务端示例处理器直接反序列化请求。pickle 可以在加载对象时执行代码，因此此协议只适合双方内容均受信任的 Python 环境；它不构成面向不可信客户端的安全数据格式。[Python pickle 文档](https://docs.python.org/3/library/pickle.html)

传输中没有身份认证、消息完整性校验或加密。返回动作还会进入机器人执行路径，服务发现成功并不建立动作来源的可信性。

服务端启动前调用 `check_and_free_port()`。如果端口占用，它会用 `lsof` 查找进程并发送强制终止信号。这是当前启动流程的实际副作用，不能把运行该示例当作单纯的端口探测。
