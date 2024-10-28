# ECE759

UW-Madison 24Fall

## QuickStart

1. Install [Docker](https://docs.docker.com/engine/install/) on your local machine (MacOS/Windows/Linux).

2. Git clone the repo in local dicretory:

   ```
   git clone git@github.com:starFalll/ECE759.git
   ```

3. Build Dockerfile:

   ```
   docker build -t ece759 .
   ```

4. Run Docker container's interactive shell in the ECE759 dicretory:

   ```
   docker container run -it -v "$(pwd)/source:/workspace/source" -v "$(pwd)/photos:/workspace/photos" ece759
   ```

   `-v` will mount your local machine's directory to the container's /workspace/* directory.

5. Add new CPP `.h` and `.cpp` files below `source` directory, implementing your logic here. In your code, save the photo results to the directory that below `photos` (Please `mkdir` your own directory below `photos`).

6. Compile source code in Docker container, for example, compile `homography`:

   ```
   root@xxx:/workspace/source# g++ -fopenmp -o homography_app homography.cpp homography.h `pkg-config --cflags --libs opencv4`
   ```

7. Review the results and debug the issues.

8. Exit the Docker:

   ```
   root@xxx:/workspace# exit
   ```

9. Git add, commit and push to your own branch, ask @Ethan to review and merge.

