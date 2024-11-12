#!/bin/bash

# Compile the RANSAC project with C++17

# Set compiler
CXX=g++

# Set source files
SOURCES="ransac.cpp helper.cpp backwardWarpImg.cpp blendImagePair.cpp homography.cpp"

# Set output binary name
OUTPUT="ransac_program"

# Compile with C++17, linking OpenCV
$CXX -std=c++17 $SOURCES -o $OUTPUT `pkg-config --cflags --libs opencv4`

# Check compilation result
if [ $? -eq 0 ]; then
    echo "Compilation successful. Run ./$OUTPUT to execute the program."
else
    echo "Compilation failed. Please check the errors above."
fi