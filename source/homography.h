#ifndef HOMOGRAPHY_H
#define HOMOGRAPHY_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

Eigen::Matrix3d computeHomography(const std::vector<Eigen::Vector2d>& src_pts, const std::vector<Eigen::Vector2d>& dest_pts);
std::vector<Eigen::Vector2d> applyHomography(const Eigen::Matrix3d& H, const std::vector<Eigen::Vector2d>& src_pts);
cv::Mat showCorrespondence(const cv::Mat& img1, const cv::Mat& img2, const std::vector<Eigen::Vector2d>& pts1, const std::vector<Eigen::Vector2d>& pts2);


#endif