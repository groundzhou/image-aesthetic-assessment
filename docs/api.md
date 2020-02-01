---
title: API设计文档
---

# 1. 图片评价接口

接口描述：发送openid、图片，返回图片评分结果

URL： https://api.groundzhou.cn:5000

示例：

请求
```json
{
    "Accept": "application/json",
    "method": "POST",
    "data": {
        "image": "",
        "_openid": "123"  
  }
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
        "color": 6
    }
}
```