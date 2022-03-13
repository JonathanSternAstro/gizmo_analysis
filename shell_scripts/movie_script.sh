#!/bin/bash

# Input parameters
# $1 -- sim name, e.g. feedback_light
# $2 -- edge_on or frame_on
# $3 -- framerate, e.g. 15

module load ffmpeg

ffmpeg -framerate $3 -i ../figures/movieFrames/$1/temperature_and_pressure_movieFrame_$2_%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ../figures/movies/$1_$2.mp4


#ffmpeg -r 24 -i temperature_and_pressure_movieFrame_%00d.png -vcodec libx264 -y -an video.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
