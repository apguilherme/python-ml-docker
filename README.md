### create image
`docker build -t python-ml-docker . -f Dockerfile`

`docker build -t python-ml-docker-prod . -f Dockerfile.prod`

### remove image
`docker image rm python-ml-docker`

`docker image rm python-ml-docker-prod`

### start container
`docker run -p 5050:5050 python-ml-docker`

`docker run -p 8000:8000 python-ml-docker-prod`

### create and run
`docker build -t python-ml-docker . -f Dockerfile && docker run -p 5050:5050 python-ml-docker`

`docker build -t python-ml-docker-prod . -f Dockerfile.prod && docker run -p 8000:8000 python-ml-docker-prod`

### swagger docs
`http://127.0.0.1:5050/apidocs/`

`http://127.0.0.1:8000/apidocs/`

### debug
`docker ps`

`docker exec -it <CONTAINER ID> bash`

`ls`

`cd /etc/mod_wsgi-express-80`

`vim error_log`

---
Based on Udemy Course: 
> Deploying AI & Machine Learning Models for Business | Python: Learn to build Machine Learning, Deep Learning & NLP Models & Deploy them with Docker Containers (DevOps) (in Python)
