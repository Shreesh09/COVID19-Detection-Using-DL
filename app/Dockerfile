FROM python:3.10.12
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN useradd -m -u 1000 user
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y libgl1 
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
CMD [ "python3", "app.py" ]
