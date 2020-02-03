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
    "url": "https://api.groundzhou.cn:5000",
    "method": "POST",
    "data": {
        "openid": "123" ,
        "image": "sadfasf"    
    },
    "header": {
        "content-type": "application/json"
    }
}
```

响应
1. 成功：

```json
{
    "code": 0,
    "message": "success",
    "data": {
        "score": 7,
        "light": 9.2,
        "color": 6
    }
}
```
