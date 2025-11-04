to build the docker containter sudo docker build -t maskrcnn .

to run the docker container  

xhost +local:root

sudo docker run -it --rm   --runtime nvidia   --privileged   --network host   --ipc host   -v ~/fish_size/fishsize_maskrcnn:/workspace/host   -v /dev:/dev  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY  maskrcnn

cd ..
cd host
python3 python3 measure_fish_automatic.py 

