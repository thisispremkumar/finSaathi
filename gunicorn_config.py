# Gunicorn configuration for production deployment

bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
keepalive = 2
timeout = 30
preload_app = True