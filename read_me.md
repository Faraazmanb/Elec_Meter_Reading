## Run Docker Image:
### Demo:
Two models, last and best
`docker run metermodel python app.py a.jpg last`
`docker run metermodel python app.py a.jpg best`

### Run using local image file:
docker run -v <path to folder>:/app/data metermodel python app.py /app/data/<file name> last
docker run -v <path to folder>:/app/data metermodel python app.py /app/data/<file name> best

