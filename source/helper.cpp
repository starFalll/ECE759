#include "helper.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>> genSIFTMatches(
    const cv::Mat& img_s,
    const cv::Mat& img_d) {

    // Convert images to grayscale
    cv::Mat gray_s, gray_d;
    cv::cvtColor(img_s, gray_s, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_d, gray_d, cv::COLOR_BGR2GRAY);

    // Detect and compute SIFT features
    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_s, keypoints_d;
    cv::Mat descriptors_s, descriptors_d;

    sift->detectAndCompute(gray_s, cv::noArray(), keypoints_s, descriptors_s);
    sift->detectAndCompute(gray_d, cv::noArray(), keypoints_d, descriptors_d);

    // Match descriptors using BFMatcher with cross-check
    cv::BFMatcher matcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_s, descriptors_d, matches);

    // Extract the locations of matched keypoints
    std::vector<Eigen::Vector2d> xs, xd;
    for (const auto& match : matches) {
        const cv::KeyPoint& kp_s = keypoints_s[match.queryIdx];
        const cv::KeyPoint& kp_d = keypoints_d[match.trainIdx];
        xs.emplace_back(kp_s.pt.x, kp_s.pt.y);
        xd.emplace_back(kp_d.pt.x, kp_d.pt.y);
    }

    return {xs, xd};
}