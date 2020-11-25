FROM python:3.7.7
RUN apt-get update


#RUN pip install --upgrade pip


WORKDIR /semantic_search
COPY . /semantic_search


RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "./app.py"]