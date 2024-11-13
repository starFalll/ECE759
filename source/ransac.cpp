#include "ransac.h"
#include "homography.h"
#include "helper.h"
#include "backwardWarpImg.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <random>

std::pair<std::vector<bool>, Eigen::Matrix3d> runRANSAC(
    const std::vector<Eigen::Vector2d>& src_pt,
    const std::vector<Eigen::Vector2d>& dest_pt,
    int ransac_n,
    double eps) {
    
    std::vector<int> best_point;
    Eigen::Matrix3d best_H;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, src_pt.size() - 1);

    for (int i = 0; i < ransac_n; ++i) {
        // Randomly select 4 points
        std::vector<int> idx(4);
        for (int& id : idx) {
            id = dis(gen);
        }

        std::vector<Eigen::Vector2d> src(4);
        std::vector<Eigen::Vector2d> dest(4);
        for (int j = 0; j < 4; ++j) {
            src[j] = src_pt[idx[j]];
            dest[j] = dest_pt[idx[j]];
        }

        // Compute homography
        Eigen::Matrix3d H = computeHomography(src, dest);

        // Apply homography
        std::vector<Eigen::Vector2d> dest_hat = applyHomography(H, src_pt);

        // Find inliers
        std::vector<int> valid_point;
        for (size_t j = 0; j < dest_hat.size(); ++j) {
            double distance = (dest_hat[j] - dest_pt[j]).norm();
            if (distance < eps) {
                valid_point.push_back(j);
            }
        }

        // Update best points and homography if current iteration is better
        if (valid_point.size() > best_point.size()) {
            best_point = valid_point;
            best_H = H;
        }

        if (valid_point.size() == src_pt.size()) {
            break;
        }
    }

    // Create inliers mask
    std::vector<bool> inliers_mask(src_pt.size(), false);
    for (int idx : best_point) {
        inliers_mask[idx] = true;
    }

    return {inliers_mask, best_H};
}

int main() {
    // Load source and destination images
    cv::Mat img_src = cv::imread("../photos/data/mountain_left.jpg");
    cv::Mat img_dst = cv::imread("../photos/data/mountain_center.jpg");

    if (img_src.empty() || img_dst.empty()) {
        std::cerr << "Error: Could not load images." << std::endl;
        return -1;
    }

    // Generate SIFT Matches
    auto [xs, xd] = genSIFTMatches(img_src, img_dst);

    // Show correspondence before RANSAC
    cv::Mat before_img = showCorrespondence(img_src, img_dst, xs, xd);
    cv::imwrite("../photos/data/before_ransac.png", before_img);

    // RANSAC parameters
    int ransac_n = 4000; // Max number of iterations
    double ransac_eps = 2.0; // Acceptable alignment error

    // Run RANSAC to reject outliers
    auto [inliers_mask, _] = runRANSAC(xs, xd, ransac_n, ransac_eps);

    // Filter inliers
    std::vector<Eigen::Vector2d> xs_inliers, xd_inliers;
    for (size_t i = 0; i < inliers_mask.size(); ++i) {
        if (inliers_mask[i]) {
            xs_inliers.push_back(xs[i]);
            xd_inliers.push_back(xd[i]);
        }
    }

    // Show correspondence after RANSAC
    cv::Mat after_img = showCorrespondence(img_src, img_dst, xs_inliers, xd_inliers);
    cv::imwrite("../photos/data/after_ransac.png", after_img);

    return 0;
}