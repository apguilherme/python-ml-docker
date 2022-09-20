FROM continuumio/anaconda3
COPY . /python-ml-docker
EXPOSE 5050
WORKDIR /python-ml-docker
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD python api.py
