FROM jupyter/datascience-notebook

# RUN apk --update-cache \
#     add musl \
#     linux-headers \
#     gcc \
#     g++ \
#     make \
#     gfortran \
#     openblas-dev \
#     python3 \
#     python3-dev \
#     freetype-dev \
#     libpng-dev


# RUN pip3 install --upgrade pip 

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . /app
WORKDIR /app

CMD tail -f /dev/null
