#include "backwardWarpImg.h"
#include "common.h"
#include <omp.h>

std::pair<cv::Mat, cv::Mat> backwardWarpImg(const cv::Mat& src_img, const Eigen::Matrix3d& destToSrc_H, const cv::Size& canvas_shape) {
    // Input arguments: {src_img, destToSrc_H, canvas_shape}.
    // src_img is the source image,3-channel float32 (CV_32FC3) matrix, range [0.0f, 1.0f] with size (width, height, 3)
    // destToSrc_H: inverse of H_3x3 
    // canvas_shape is the shape of canvas with (width, height)
    
    // Output: {dest_mask, dest_img}. 
    // dest_mask is a uint8 (CV_8U) binary matrix, the values are 0 or 1
    // dest_img is a 3-channel float32 (CV_32FC3) matrix, range 0.0 - 1.0, are the warpped source image


    // Validate Inputs
    // 1. Check if src_img is empty
    if (src_img.empty()) {
        throw std::invalid_argument("Error: src_img is empty.");
    }

    // 2. Check src_img type
    if (src_img.type() != CV_32FC3) {
        throw std::invalid_argument("Error: src_img must be of type CV_32FC3 (3-channel, float32).");
    }

    // 3. Check src_img value range (optional, for debugging)
    double minVal, maxVal;
    cv::minMaxLoc(src_img, &minVal, &maxVal);
    if (minVal < 0.0 || maxVal > 1.0) {
        throw std::invalid_argument("Error: src_img values must be in range [0.0, 1.0].");
    }

    // 4. Check canvas_shape dimensions
    if (canvas_shape.width <= 0 || canvas_shape.height <= 0) {
        throw std::invalid_argument("Error: canvas_shape dimensions must be positive.");
    }

    // 5. Check destToSrc_H validity (optional)
    if (destToSrc_H.rows() != 3 || destToSrc_H.cols() != 3) {
        throw std::invalid_argument("Error: destToSrc_H must be a 3x3 matrix.");
    }



    // initialize dest_img and mask
    cv::Mat dest_img = cv::Mat::zeros(canvas_shape, CV_32FC3);  // dest_img, float32, 3 channels, range 0.0 - 1.0
    cv::Mat dest_mask = cv::Mat::zeros(canvas_shape, CV_8U);    // mask, uint8, values: 0 or 1

    int height_src = src_img.rows;
    int width_src = src_img.cols;

    const float* src_ptr = src_img.ptr<float>();
    float* dest_ptr = dest_img.ptr<float>();
    uchar* mask_ptr = dest_mask.ptr<uchar>();

    // OpenMP
    #pragma omp parallel for OMP_SCHEDULE(FOR_SCHEDULE_TYPE, CHUNKS_PER_THREAD) collapse(2)
    for (int y = 0; y < canvas_shape.height; ++y) {
        for (int x = 0; x < canvas_shape.width; ++x) {
            // Use the homography matrix to calculate corresponding points in the source image
            Eigen::Vector3d src_pt(x, y, 1.0);
            Eigen::Vector3d src_coords = destToSrc_H * src_pt;

            // Normalization using the bias_z of (src_x, src_y, bias_z) to obtain normalized (src_x, src_y) coordinates
            double src_x = src_coords(0) / src_coords(2);
            double src_y = src_coords(1) / src_coords(2);

            // check if the calculated src_x and src_y are within the source image
            int src_x_int = static_cast<int>(std::round(src_x));
            int src_y_int = static_cast<int>(std::round(src_y));

            if (src_x_int >= 0 && src_x_int < width_src && src_y_int >= 0 && src_y_int < height_src) {
                int src_idx = (src_y_int * width_src + src_x_int) * 3; // row order index: (row_index * image_width + column_index)*3, multiple 3 is because each pixel has 3 channels
                int dest_idx = (y * canvas_shape.width + x) * 3;

                dest_ptr[dest_idx] = src_ptr[src_idx];
                dest_ptr[dest_idx + 1] = src_ptr[src_idx + 1];
                dest_ptr[dest_idx + 2] = src_ptr[src_idx + 2];
                mask_ptr[y * canvas_shape.width + x] = 1;
            }
        }
    }

    return {dest_mask, dest_img};
}


/////////////////////////////////////////////////////////////
// Below is the main() funtion for testing the backwardWarpImg.cpp. 
// If you just want to test the backwardWarpImg.cpp, please copy the main function code to a separate main function file outside 
// Compile： g++ -fopenmp -o test_program main.cpp backwardWarpImg.cpp homography.cpp `pkg-config --cflags --libs opencv4` -I /usr/include/eigen3
// execute:  ./test_program
/////////////////////////////////////////////////////////////

// #include <opencv2/opencv.hpp>
// #include "backwardWarpImg.h"  //
// #include "homography.h"


// int main() {
//     // Loading background and foreground images
//     cv::Mat bg_img = cv::imread("../photos/backwardWarpImg_data/Osaka.png");
//     cv::Mat portrait_img = cv::imread("../photos/backwardWarpImg_data/portrait_small.png");

//     if (bg_img.empty() || portrait_img.empty()) {
//         std::cerr << "Cannot load image, please check the path." << std::endl;
//         return -1;
//     }

//     // Convert the image to a floating point value between [0, 1]
//     bg_img.convertTo(bg_img, CV_32FC3, 1.0 / 255.0);
//     portrait_img.convertTo(portrait_img, CV_32FC3, 1.0 / 255.0);

//     // Define the source and destination points
//     std::vector<Eigen::Vector2d> src_pts = { {3, 2},{324, 2}, {3, 399}, {326, 398}};
//     std::vector<Eigen::Vector2d> dest_pts = { {101, 19}, {276, 71}, {85, 436}, {285, 424}};

//     // Compute the homography matrix H and its inverse
//     Eigen::Matrix3d H = computeHomography(src_pts, dest_pts);
//     Eigen::Matrix3d inv_H = H.inverse();

//     // Get the target canvas size
//     cv::Size dest_canvas_shape(bg_img.cols, bg_img.rows);
//     std::cout << "Destination canvas shape (width, height): " << dest_canvas_shape << std::endl;

//     // Call backwardWarpImg function
//     auto back_result = backwardWarpImg(portrait_img, inv_H, dest_canvas_shape);
//     cv::Mat mask = back_result.first;
//     cv::Mat dest_img = back_result.second;

//     // Inverse mask for background overlay (1 to 0, 0 to 1)
//     cv::Mat mask_inv = 1 - mask;
//     cv::Mat mask_inv_normalized;
//     mask_inv.convertTo(mask_inv_normalized, CV_8UC1, 255.0);  //  contert to [0, 255] 
//     cv::imwrite("../photos/backwardWarpImg_data/debug_mask_inv.png", mask_inv_normalized);

//     // expand mask to 3 channels
//     cv::Mat mask_3c;
//     cv::Mat mask_channels[] = {mask_inv, mask_inv, mask_inv };
//     cv::merge(mask_channels, 3, mask_3c);
//     mask_3c.convertTo(mask_3c, CV_32FC3);

//     // save dest_img
//     cv::Mat dest_img_normalized;
//     dest_img.convertTo(dest_img_normalized, CV_8UC3, 255.0);
//     cv::imwrite("../photos/backwardWarpImg_data/debug_dest_img.png", dest_img_normalized);

//     // save mask
//     cv::Mat mask_normalized;
//     mask.convertTo(mask_normalized, CV_8UC1, 255.0);
//     cv::imwrite("../photos/backwardWarpImg_data/debug_mask.png", mask_normalized);

//     // create the merged image
//     cv::Mat result = bg_img.mul(mask_3c) + dest_img;
//     result.convertTo(result, CV_8UC3, 255.0);
//     cv::imwrite("../photos/backwardWarpImg_data/Van_Gogh_in_Osaka.png", result);

//     return 0;
// }





