#ifndef BLEND_IMAGE_PAIR_H
#define BLEND_IMAGE_PAIR_H

#include <opencv2/opencv.hpp>
#include <string>

cv::Mat blendImagePair(const cv::Mat& img1, const cv::Mat& mask1, const cv::Mat& img2, const cv::Mat& mask2, const std::string& mode);

#endif // BLEND_IMAGE_PAIR_H
