#ifndef STITCH_IMG_H
#define STITCH_IMG_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense> 

cv::Mat stitchImg(const std::vector<cv::Mat>& imgs);

#endif
