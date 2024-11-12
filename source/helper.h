#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>> genSIFTMatches(
    const cv::Mat& img_s,
    const cv::Mat& img_d);

#endif