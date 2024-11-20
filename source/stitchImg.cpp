#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include "stitchImg.h"
#include "homography.h"
#include "helper.h"
#include "ransac.h"
#include "blendImagePair.h"
#include "backwardWarpImg.h"

Eigen::Matrix3d transferHomography(const Eigen::Matrix3d& H, double tx, double ty) {
    Eigen::Matrix3d transfer_matrix = Eigen::Matrix3d::Identity();
    transfer_matrix(0, 2) = tx;
    transfer_matrix(1, 2) = ty;
    return transfer_matrix * H;
}

cv::Mat stitchImg(const std::vector<cv::Mat>& imgs) {
    constexpr int dimension = 255;
    cv::Mat left = imgs[0].clone();

    // Replace black pixels (value 0) in left with 1/dimension
    #pragma omp parallel for collapse(3)
    for (int y = 0; y < left.rows; ++y) {
        for (int x = 0; x < left.cols; ++x) {
            for (int c = 0; c < left.channels(); ++c) {
                if (left.at<cv::Vec3b>(y, x)[c] == 0) {
                    left.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(1.0 / dimension);
                }
            }
        }
    }

    for (size_t idx = 1; idx < imgs.size(); ++idx) {
        cv::Mat right = imgs[idx].clone();

        #pragma omp parallel for collapse(3)
        for (int y = 0; y < right.rows; ++y) {
            for (int x = 0; x < right.cols; ++x) {
                for (int c = 0; c < right.channels(); ++c) {
                    if (right.at<cv::Vec3b>(y, x)[c] == 0) {
                        right.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(1.0 / dimension);
                    }
                }
            }
        }

        // 1. first get the Homography after denoising
        auto [xs, xd] = genSIFTMatches(right, left);

        int ransac_n = 2000;
        double ransac_eps = 10.0;
        auto [_, H] = runRANSAC(xs, xd, ransac_n, ransac_eps);

        // 2. pick four corners (two functions: 1. compute the size of warp img; 2. compute the update Homography)
        std::vector<Eigen::Vector2d> right_corners = {
            {0, 0}, {right.cols - 1, 0}, {right.cols - 1, right.rows - 1}, {0, right.rows - 1}};
        auto left_corners = applyHomography(H, right_corners);

        double origin_x = std::min_element(left_corners.begin(), left_corners.end(),
                                           [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) { return a.x() < b.x(); })->x();
        double origin_y = std::min_element(left_corners.begin(), left_corners.end(),
                                           [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) { return a.y() < b.y(); })->y();

        double new_origin_x = std::max(0.0, -origin_x);
        double new_origin_y = std::max(0.0, -origin_y);

        int new_x_len = static_cast<int>(std::ceil(std::max_element(left_corners.begin(), left_corners.end(),
                                                                     [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) { return a.x() < b.x(); })->x() +
                                                    new_origin_x));
        int new_y_len = static_cast<int>(std::ceil(std::max_element(left_corners.begin(), left_corners.end(),
                                                                     [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) { return a.y() < b.y(); })->y() +
                                                    new_origin_y));

        H = transferHomography(H, new_origin_x, new_origin_y);

        cv::Size dest_canvas_shape(new_x_len, new_y_len);
        cv::Mat curr_canvas(dest_canvas_shape, left.type(), cv::Scalar::all(0));
        left.copyTo(curr_canvas(cv::Rect(static_cast<int>(new_origin_x), static_cast<int>(new_origin_y), left.cols, left.rows)));

        cv::Mat mask = (curr_canvas != 0);

        auto [dest_img, dest_mask] = backwardWarpImg(right, H.inverse(), dest_canvas_shape);

        left = blendImagePair(curr_canvas, mask, dest_img, dest_mask, "blend");

        #pragma omp parallel for collapse(3)
        for (int y = 0; y < left.rows; ++y) {
            for (int x = 0; x < left.cols; ++x) {
                for (int c = 0; c < left.channels(); ++c) {
                    if (left.at<cv::Vec3b>(y, x)[c] == static_cast<unsigned char>(1.0 / dimension)) {
                        left.at<cv::Vec3b>(y, x)[c] = 0;
                    }
                }
            }
        }
    }

    return left;
}

int main() {
    // Load example images
    cv::Mat img_center = cv::imread("../photos/data/mountain_center.jpg");
    cv::Mat img_left = cv::imread("../photos/data/mountain_left.jpg");
    cv::Mat img_right = cv::imread("../photos/data/mountain_right.jpg");

    if (img_center.empty() || img_left.empty() || img_right.empty()) {
        std::cerr << "Could not load images." << std::endl;
        return -1;
    }
    std::vector<cv::Mat> imgs;
    imgs.push_back(img_center);
    imgs.push_back(img_left);
    imgs.push_back(img_right);

    // stitch images
    cv::Mat result = stitchImg(imgs);

    // Save the result
    cv::imwrite("../photos/data/stitched_mountain.png", result);

    return 0;
}