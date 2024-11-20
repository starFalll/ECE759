#!/bin/bash

# Compile project with C++17

DEBUG=0

# Set compiler
CXX=g++

# Set source files
SOURCES="stitchImg.cpp ransac.cpp helper.cpp backwardWarpImg.cpp blendImagePair.cpp homography.cpp"

# Set output binary name
OUTPUT="stitch_image"

# Compile with C++17, linking OpenCV
if [ "$DEBUG" -eq 0 ]; then
    echo "Compilation with O2 optimization."
    $CXX -std=c++17 -O2 $SOURCES -o $OUTPUT `pkg-config --cflags --libs opencv4` -fopenmp
else
    echo "Compilation with debug info."
    $CXX -std=c++17 -g $SOURCES -o $OUTPUT `pkg-config --cflags --libs opencv4` -fopenmp
fi

# Check compilation result
if [ $? -eq 0 ]; then
    echo "Compilation successful. Run ./$OUTPUT thread_num to execute the program."
else
    echo "Compilation failed. Please check the errors above."
fi