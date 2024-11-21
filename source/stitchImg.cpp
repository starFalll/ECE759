#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <string>
#include <cstdlib>
#include "stitchImg.h"
#include "homography.h"
#include "helper.h"
#include "ransac.h"
#include "blendImagePair.h"
#include "backwardWarpImg.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

void PrintMat(const cv::Mat& img, std::string name)
{
    std::cout<<name<<": size: "<<img.size()<<" channel:"<<img.channels()<<" type:"<<img.type()<<std::endl;
}

Eigen::Matrix3d transferHomography(const Eigen::Matrix3d& H, double tx, double ty) {
    Eigen::Matrix3d transfer_matrix = Eigen::Matrix3d::Identity();
    transfer_matrix(0, 2) = tx;
    transfer_matrix(1, 2) = ty;
    return transfer_matrix * H;
}

cv::Mat stitchImg(const std::vector<cv::Mat>& imgs) {
    constexpr int dimension = 255;
    cv::Mat left = imgs[0].clone();

    for (size_t idx = 1; idx < imgs.size(); ++idx) {
        cv::Mat right = imgs[idx].clone();


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
        new_x_len = std::max(new_x_len, left.cols + int(new_origin_x));                          
        int new_y_len = static_cast<int>(std::ceil(std::max_element(left_corners.begin(), left_corners.end(),
                                                                     [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) { return a.y() < b.y(); })->y() +
                                                    new_origin_y));
        new_y_len = std::max(new_y_len, left.rows + int(new_origin_y));
        H = transferHomography(H, new_origin_x, new_origin_y);

        cv::Size dest_canvas_shape(new_x_len, new_y_len);
        cv::Mat curr_canvas(dest_canvas_shape, left.type(), cv::Scalar::all(0));
        left.copyTo(curr_canvas(cv::Rect(static_cast<int>(new_origin_x), static_cast<int>(new_origin_y), left.cols, left.rows)));


        cv::Mat channels[3];
        cv::split(curr_canvas, channels);
        cv::Mat mask = (channels[0] != 0) | (channels[1] != 0) | (channels[2] != 0);
        

        right.convertTo(right, CV_32FC3, 1.0 / 255.0);
        auto [dest_mask, dest_img] = backwardWarpImg(right, H.inverse(), dest_canvas_shape);

        // Normalize the image to the range [0, 1] and convert to floating point
        curr_canvas.convertTo(curr_canvas, CV_32F, 1.0 / 255.0);
        left = blendImagePair(curr_canvas, mask, dest_img, dest_mask, "blend");
        left.convertTo(left, CV_8U, 255.0);
    }

    return left;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "please run commond: ./stitch_image thread_num " << std::endl;
        return -1;
    }
    int thread_num = atoi(argv[1]);
    // Load images
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

    // set thread_num
    omp_set_num_threads(thread_num);

    auto start_time = high_resolution_clock::now();
    // stitch images
    cv::Mat result = stitchImg(imgs);
    auto end_time = high_resolution_clock::now();
    auto duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end_time - start_time);

    // std::cout<<"Success! Total time:"<< duration_sec.count()<<"ms"<<std::endl;
    std::cout<<duration_sec.count()<<std::endl;
    

    // Save the result
    cv::imwrite("../photos/data/stitched_mountain.png", result);

    // if (argc != 2) {
    //     std::cout << "please run commond: ./stitch_image thread_num " << std::endl;
    //     return -1;
    // }
    // int thread_num = atoi(argv[1]);

    // std::vector<cv::Mat> imgs;
    // for (int i = 2; i >= 0; i--) {
    //     imgs.emplace_back(cv::imread("../photos/data/input/1114008" + std::to_string(i) + "_l.PNG"));
    // }
    // for (int i = 3; i < 6; i++) {
    //     imgs.emplace_back(cv::imread("../photos/data/input/1114008" + std::to_string(i) + "_l.PNG"));
    // } 

    // // set thread_num
    // omp_set_num_threads(thread_num);

    // auto start_time = high_resolution_clock::now();
    // // stitch images
    // cv::Mat result = stitchImg(imgs);
    // auto end_time = high_resolution_clock::now();
    // auto duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end_time - start_time);

    // // std::cout<<"Success! Total time:"<< duration_sec.count()<<"ms"<<std::endl;
    // std::cout<<duration_sec.count()<<std::endl;
    // // Save the result
    // cv::imwrite("../photos/data/stitched_school.png", result);


    return 0;
}