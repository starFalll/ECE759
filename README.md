# ECE759

UW-Madison 24Fall

## QuickStart

### Install

1. Install [Docker](https://docs.docker.com/engine/install/) on your local machine (MacOS/Windows/Linux).

2. Git clone the repo in local dicretory:

   ```
   git clone git@github.com:starFalll/ECE759.git
   ```

3. Build Dockerfile:

   ```
   docker build -t ece759 .
   ```

We can always use the same docker image without building the docker again.

### Run

1. Run Docker container's interactive shell in the ECE759 dicretory:

   ```
   docker container run -it -v "$(pwd)/source:/workspace/source" -v "$(pwd)/photos:/workspace/photos" ece759
   ```

   `-v` will mount your local machine's directory to the container's /workspace/* directory.

2. Add new CPP `.h` and `.cpp` files below `source` directory, implementing your logic here. In your code, save the photo results to the directory that below `photos` (Please `mkdir` your own directory below `photos`).

3. Compile source code in Docker container, for example, compile `stitch_image`:

   In `build.sh`, we can choose to generate debug or release version:

   ```
   Debug=0  # 0:release, 1:debug
   ```

   The default is release version, then we run commond:

   ```shell
   root@xxx:/workspace/source# sh build.sh
   Compilation with O2 optimization.
   Compilation successful. Run ./stitch_image to execute the program.
   root@xxx:/workspace/source# ./stitch_image
   ```

4. Review the results in `photos/data/stitched_mountain.png`, and debug the issues:

5. Exit the Docker:

   ```
   root@xxx:/workspace# exit
   ```

6. Git add, commit and push to your own branch, ask @Ethan to review and merge.

### Others

1. Supplymentary Jidong's comments:
   In the blendImagePair.cpp and backwardWarpImg.cpp files, there are tips for how to compile these two files, and you can also check the input and output data format with the attached commented example main function.
