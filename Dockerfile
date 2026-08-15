# -----------------------------
# Creation of a Python env for experimentation with correct data
# Original repo: https://github.com/Gci04/GANs_for_Network_Intrusion_Data
#
# Build/run commands (from the folder containing this Dockerfile):
#
#   docker build -t gan-env .
#   docker run --rm -it gan-env bash
#
# Inside the container:
#   cd /app/GANs_for_Network_Intrusion_Data
#   python -c "import numpy, pandas, matplotlib, keras, tensorflow, seaborn; print('ok')"
#
#   Navigate to desired Data Generative Model 'cd NSL-KDD' or 'cd UNSW-NB15', then train and 
#   test the model by running train and test scripts : python train.py first, then python test.py.
# -----------------------------

FROM python:3.7-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/EduSanSou/GANs_for_Network_Intrusion_Data.git

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      numpy==1.19.5 \
      pandas==1.2.5 \
      matplotlib==3.4.3 \
      keras==2.2.4 \
      tensorflow==2.4.4 \
      pandas-profiling==3.1.0 \
      tabulate==0.8.10 \
      imbalanced-learn==0.8.0 \
      category_encoders==2.2.2 \
      scikit-learn==0.24.2 \
      seaborn==0.11.2

CMD ["bash"]

