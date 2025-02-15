user=$(whoami)
image_name=eml_exercise
image_id=$(docker images | grep $image_name | awk '{print $3}')

dir=/home/$user
workdir=/home/$user/Embedded_Machine_Learning_24ss_Group5/exercise/ex05

docker run --gpus all --rm -it -w $workdir -v $dir:$dir  --shm-size=10g --ulimit memlock=1 --ulimit stack=67108864 $image_id
