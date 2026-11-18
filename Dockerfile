FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nmap git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/RickSanchezC18/Signal_Translate_RP3MB.git .

RUN pip3 install -r requirements.txt

CMD ["python3", "main_traductor.py"]   