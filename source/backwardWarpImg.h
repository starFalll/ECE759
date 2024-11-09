#ifndef BACKWARD_WARP_IMG_H
#define BACKWARD_WARP_IMG_H

#include <opencv2/opencv.hpp>
#include <utility>
#include <Eigen/Dense> 

std::pair<cv::Mat, cv::Mat> backwardWarpImg(const cv::Mat& src_img, const Eigen::Matrix3d& destToSrc_H, const cv::Size& canvas_shape);

#endif // BACKWARD_WARP_IMG_H
