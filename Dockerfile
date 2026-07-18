# 使用Python官方镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露Gradio默认端口
EXPOSE 7860

# 启动Gradio应用（假设app.py是入口）
CMD ["python", "app.py"]
