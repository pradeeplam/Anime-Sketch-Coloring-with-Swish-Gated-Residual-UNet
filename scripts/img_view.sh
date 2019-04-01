#!/bin/sh

# Navigate to your log/${exp_name}/images directory and run this script
# from there. 
# * the program feh needs to be installed

rm target.jpg
latest=$(ls -t | grep --color=never -v "img_view.sh" | head -n1)
ln -s $latest target.jpg
echo "Latest: $latest @ $(date)"

# feh to preview the image
feh --scale-down --reload 10 target.jpg &
sleep 10

while true; do
    rm target.jpg
    latest=$(ls -t | head -n1)
    ln -s $latest target.jpg
    echo "Latest: $latest @ $(date)"
    sleep 10
done;
