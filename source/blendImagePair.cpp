#include "blendImagePair.h"
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat blendImagePair(const cv::Mat& img1, const cv::Mat& mask1, const cv::Mat& img2, const cv::Mat& mask2, const std::string& mode) {
    // Input: "img1" and "img2" (normalized CV_32F, 3 channel, range 0.0-1.0); "mask1" and "mask2" (normalized CV_32F, one channel 0.0-1.0)
    // Output: "out_img" (CV_32F, 3 channels, range 0.0-0.1)
    cv::Mat out_img;
    cv::Mat mask1_8U, mask2_8U;

    // Convert the mask to CV_8U and normalize to 0-255 (in python it is 0-1)
    mask1_8U = mask1 > 0;
    mask2_8U = mask2 > 0;
    mask1_8U.convertTo(mask1_8U, CV_8U);
    mask2_8U.convertTo(mask2_8U, CV_8U);

    // // Output debug information
    // std::cout << "img1 size: " << img1.size() << ", channels: " << img1.channels() << std::endl;
    // std::cout << "img2 size: " << img2.size() << ", channels: " << img2.channels() << std::endl;
    // std::cout << "mask1 size: " << mask1.size() << ", channels: " << mask1.channels() << std::endl;
    // std::cout << "mask2 size: " << mask2.size() << ", channels: " << mask2.channels() << std::endl;

    if (mode == "overlay") {
        // use mask to overlay img
        out_img = img1.clone();
        img2.copyTo(out_img, mask2_8U);  // Copy img2 to the area where mask2_8u is non-zero
    } else if (mode == "blend") {
        // Use distance transform for smooth blending
        cv::Mat dist_transform1, dist_transform2;

        cv::distanceTransform(mask1_8U, dist_transform1, cv::DIST_L2, 3);
        cv::distanceTransform(mask2_8U, dist_transform2, cv::DIST_L2, 3);

        dist_transform1 /= cv::norm(dist_transform1, cv::NORM_INF);
        dist_transform2 /= cv::norm(dist_transform2, cv::NORM_INF);

        // Extended distance transform results in three channels
        cv::Mat dist_transform1_3c, dist_transform2_3c;
        cv::merge(std::vector<cv::Mat>{dist_transform1, dist_transform1, dist_transform1}, dist_transform1_3c);
        cv::merge(std::vector<cv::Mat>{dist_transform2, dist_transform2, dist_transform2}, dist_transform2_3c);

        cv::Mat blend_weights = dist_transform1_3c + dist_transform2_3c;
        blend_weights.setTo(1, blend_weights == 0);  // Avoid division by 0

        // Calculate the blended image using weighted average
        out_img = (dist_transform1_3c.mul(img1) + dist_transform2_3c.mul(img2)) / blend_weights;
    }

    return out_img; 
}

/////////////////////////////////////////////////////////////
// Below is the main() funtion for testing the blendImagePair.cpp
// If you just want to test the blendImagePair.cpp, please copy the main function code to a separate main function file outside 
// Compile： g++ -o blendImagePair main.cpp blendImagePair.cpp `pkg-config --cflags --libs opencv4`
// execute:  ./blendImagePair
/////////////////////////////////////////////////////////////

// #include <opencv2/opencv.hpp>
// #include "blendImagePair.h"
// #include <iostream>

// int main() {
//     // Loading an image with an alpha channel
//     cv::Mat fish = cv::imread("../photos/blendImagePair/escher_fish.png", cv::IMREAD_UNCHANGED);
//     cv::Mat horse = cv::imread("../photos/blendImagePair/escher_horsemen.png", cv::IMREAD_UNCHANGED);

//     if (fish.empty() || horse.empty()) {
//         std::cerr << "Error: Could not load images." << std::endl;
//         return -1;
//     }

//     // Separate RGB and alpha channels (as masks)
//     std::vector<cv::Mat> fish_channels, horse_channels;
//     cv::split(fish, fish_channels);
//     cv::split(horse, horse_channels);

//     cv::Mat fish_img, horse_img;
//     cv::Mat fish_mask = fish_channels[3];
//     cv::Mat horse_mask = horse_channels[3];
//     cv::merge(std::vector<cv::Mat>{fish_channels[0], fish_channels[1], fish_channels[2]}, fish_img);
//     cv::merge(std::vector<cv::Mat>{horse_channels[0], horse_channels[1], horse_channels[2]}, horse_img);

//     // Normalize the image to the range [0, 1] and convert to floating point
//     fish_img.convertTo(fish_img, CV_32F, 1.0 / 255.0);
//     horse_img.convertTo(horse_img, CV_32F, 1.0 / 255.0);
//     fish_mask.convertTo(fish_mask, CV_32F, 1.0 / 255.0);
//     horse_mask.convertTo(horse_mask, CV_32F, 1.0 / 255.0);

//     // Testing 'blend' Mode
//     cv::Mat blended_result = blendImagePair(fish_img, fish_mask, horse_img, horse_mask, "blend");
//     cv::imwrite("../photos/blendImagePair/blended_result.png", blended_result * 255);

//     // Testing 'overlay' Mode
//     cv::Mat overlay_result = blendImagePair(fish_img, fish_mask, horse_img, horse_mask, "overlay");
//     cv::imwrite("../photos/blendImagePair/overlay_result.png", overlay_result * 255);

//     std::cout << "Images saved successfully in the outputs folder." << std::endl;
//     return 0;
// }