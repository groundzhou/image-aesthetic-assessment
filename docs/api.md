---
title: API设计文档
---

接口描述：

URL： https://api.groundzhou.cn:5000

示例：

请求
```json
{
    Accept: 'application/json',
    method: 'GET'
}
```

响应
```json
{
    "code": 0,
    "message": "OK",
    "data": {
        "score": 7,
        "light": 9.2,
        "color": 6,
    }
}
```