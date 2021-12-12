FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel


RUN apt-get update
RUN apt-get install -y git python3-pip ffmpeg libsm6 libxext6

RUN pip install --no-cache-dir --upgrade pip

RUN mkdir "/EmIm"
WORKDIR "/EmIm"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# set the working directory in the container
# copy the dependencies file to the working directory
COPY setup.py .
# copy the content of the local src directory to the working directory
COPY ./src/ ./src

RUN pip install .


RUN ls -la

CMD [ "python", "src/trainers/emim_train.py" ]