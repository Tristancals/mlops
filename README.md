


monitoring


 sudo docker run -p 9090:9090 -v ./prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

```yml
global:
  scrape_interval:     15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: SwagDigitClassifier3000
    static_configs:
      - targets: ['172.17.0.1:8000']
```

 sudo docker run -d --name grafana -p 3000:3000 grafana/grafana