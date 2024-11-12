#ifndef RANSAC_H
#define RANSAC_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

std::pair<std::vector<bool>, Eigen::Matrix3d> runRANSAC(
    const std::vector<Eigen::Vector2d>& src_pt,
    const std::vector<Eigen::Vector2d>& dest_pt,
    int ransac_n,
    double eps);

#endif // RANSAC_H