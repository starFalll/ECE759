#include "blendImagePair.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>

cv::Mat blendImagePair(const cv::Mat& img1, const cv::Mat& mask1, const cv::Mat& img2, const cv::Mat& mask2, const std::string& mode) {
    // Input: "img1" and "img2" (normalized CV_32FC3, 3 channel, range 0.0-1.0); 
    // Input: "mask1" and "mask2" (binary mask, CV_8U, one channel, range [0,1] and [0,255] are both ok, since it will be automatically normalized below)
    // Output: "out_img" (CV_32FC3, 3 channels, range 0.0-1.0)

    // *** Validate Inputs ***
    // 1. Check if inputs are empty
    if (img1.empty() || img2.empty() || mask1.empty() || mask2.empty()) {
        throw std::invalid_argument("Error: One or more input matrices are empty.");
    }

    // 2. Check sizes
    if (img1.size() != img2.size()) {
        throw std::invalid_argument("Error: img1 and img2 sizes do not match.");
    }
    if (mask1.size() != img1.size() || mask2.size() != img2.size()) {
        throw std::invalid_argument("Error: Mask sizes do not match corresponding image sizes.");
    }

    // 3. Check types
    if (img1.type() != CV_32FC3 || img2.type() != CV_32FC3) {
        throw std::invalid_argument("Error: img1 and img2 must be of type CV_32FC3 (3-channel, float32).");
    }
    if (mask1.type() != CV_8U || mask2.type() != CV_8U) {
        throw std::invalid_argument("Error: mask1 and mask2 must be of type CV_8U (single-channel, 8-bit).");
    }

    // 4. Validate mode
    if (mode != "overlay" && mode != "blend") {
        throw std::invalid_argument("Error: mode must be either 'overlay' or 'blend'.");
    }

    
    // Input masks normalization (normalized binary values: 0 or 255 CV_8U)
    cv::Mat mask1_normalized = (mask1 > 0) * 255;  // Ensure binary mask
    cv::Mat mask2_normalized = (mask2 > 0) * 255;

    cv::Mat out_img;

    if (mode == "overlay") {
        // Overlay img2 onto img1 using mask2
        out_img = img1.clone();
        cv::Mat binary_mask2;
        mask2_normalized.convertTo(binary_mask2, CV_8U, 1.0 / 255);  // Convert to binary [0, 1]
        img2.copyTo(out_img, binary_mask2);
    } else if (mode == "blend") {
        // Smooth blending using distance transform
        cv::Mat dist_transform1, dist_transform2;

        // Distance Transform (these functions are already optimized in OpenCV)
        cv::distanceTransform(mask1_normalized, dist_transform1, cv::DIST_L2, 3);
        cv::distanceTransform(mask2_normalized, dist_transform2, cv::DIST_L2, 3);

        // Normalize distances
        double max1, max2;
        cv::minMaxLoc(dist_transform1, nullptr, &max1);
        cv::minMaxLoc(dist_transform2, nullptr, &max2);
        dist_transform1 /= max1 > 0 ? max1 : 1;
        dist_transform2 /= max2 > 0 ? max2 : 1;

        // Merge into 3 channels
        cv::Mat dist_transform1_3c, dist_transform2_3c;
        cv::Mat dist_channels1[3] = { dist_transform1, dist_transform1, dist_transform1 };
        cv::Mat dist_channels2[3] = { dist_transform2, dist_transform2, dist_transform2 };
        cv::merge(dist_channels1, 3, dist_transform1_3c);
        cv::merge(dist_channels2, 3, dist_transform2_3c);

        // Calculate blend weights and avoid division by zero
        cv::Mat blend_weights = dist_transform1_3c + dist_transform2_3c;
        blend_weights.setTo(1.0, blend_weights == 0);

        // Pre-allocate output image
        out_img = cv::Mat::zeros(img1.size(), img1.type());

        // Parallelize the blending operation using OpenMP
        #pragma omp parallel for
        for (int y = 0; y < img1.rows; ++y) {
            const float* img1_ptr = img1.ptr<float>(y);
            const float* img2_ptr = img2.ptr<float>(y);
            const float* weight1_ptr = dist_transform1_3c.ptr<float>(y);
            const float* weight2_ptr = dist_transform2_3c.ptr<float>(y);
            const float* blend_weight_ptr = blend_weights.ptr<float>(y);
            float* out_ptr = out_img.ptr<float>(y);

            for (int x = 0; x < img1.cols * 3; ++x) {
                out_ptr[x] = (img1_ptr[x] * weight1_ptr[x] + img2_ptr[x] * weight2_ptr[x]) / blend_weight_ptr[x];
            }
        }
    }

    return out_img;
}

/////////////////////////////////////////////////////////////
// Below is the main() funtion for testing the blendImagePair.cpp with OpenMP
// If you just want to test the blendImagePair.cpp, please copy the main function code to a separate main function file outside 
// Compile： g++ -fopenmp -o blendImagePair main.cpp blendImagePair.cpp `pkg-config --cflags --libs opencv4`
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

//     // Normalize the image to the range [0.0f, 1.0f] and convert to CV_32FC3
//     fish_img.convertTo(fish_img, CV_32FC3, 1.0 / 255.0);
//     horse_img.convertTo(horse_img, CV_32FC3, 1.0 / 255.0);

//     // CV_8U, 0 or 255, one channel. 
//     // Note:The input binary mask range [0,1] and [0,255] are both ok, since it will be automatically normalized in blendImagePair.cpp)
//     fish_mask = fish_mask > 0;
//     horse_mask = horse_mask > 0;


//     // Testing 'blend' Mode, output range [0.0f,1.0f]
//     cv::Mat blended_result = blendImagePair(fish_img, fish_mask, horse_img, horse_mask, "blend");

//     // Testing 'overlay' Mode, output range [0.0f,1.0f]
//     cv::Mat overlay_result = blendImagePair(fish_img, fish_mask, horse_img, horse_mask, "overlay");

//     cv::Mat blended_result_8u, overlay_result_8u;
//     blended_result.convertTo(blended_result_8u, CV_8UC3, 255.0);
//     overlay_result.convertTo(overlay_result_8u, CV_8UC3, 255.0);

//     cv::imwrite("../photos/blendImagePair/blended_result.png", blended_result_8u);
//     cv::imwrite("../photos/blendImagePair/overlay_result.png", overlay_result_8u);

//     std::cout << "Images saved successfully in the outputs folder." << std::endl;
//     return 0;
// }







////////////////////////////////////////////////////////////////////////
// backup blendImagePair without OpenMP: g++ -o blendImagePair main.cpp blendImagePair.cpp `pkg-config --cflags --libs opencv4`
////////////////////////////////////////////////////////////////////////

// #include "blendImagePair.h"
// #include <opencv2/opencv.hpp>
// #include <iostream>
// cv::Mat blendImagePair(const cv::Mat& img1, const cv::Mat& mask1, const cv::Mat& img2, const cv::Mat& mask2, const std::string& mode) {
//     // Input: "img1" and "img2" (normalized CV_32FC3, 3 channel, range 0.0-1.0); 
//     // Input: "mask1" and "mask2" (binary mask, CV_8U, one channel, range [0,1] and [0,255] are both ok, since it will be automatically normalized below)
//     // Output: "out_img" (CV_32FC3, 3 channels, range 0.0-1.0)
    
//     // Input masks normalization to make sure the value of mask1 and mask2 are in range of [0, 255]
//     cv::Mat mask1_normalized = (mask1 > 0) * 255;  // Ensure binary mask
//     cv::Mat mask2_normalized = (mask2 > 0) * 255;

//     cv::Mat out_img;

//     if (mode == "overlay") {
//         // use mask to overlay img
//         out_img = img1.clone();
//         img2.copyTo(out_img, mask2_normalized);  // Copy img2 to the area where mask2_normalized is non-zero
//     } else if (mode == "blend") {
//         // Use distance transform for smooth blending
//         cv::Mat dist_transform1, dist_transform2;

//         cv::distanceTransform(mask1_normalized, dist_transform1, cv::DIST_L2, 3);
//         cv::distanceTransform(mask2_normalized, dist_transform2, cv::DIST_L2, 3);

//         // Normalize distances
//         double max1, max2;
//         cv::minMaxLoc(dist_transform1, nullptr, &max1, nullptr, nullptr, mask1_normalized);
//         cv::minMaxLoc(dist_transform2, nullptr, &max2, nullptr, nullptr, mask2_normalized);
//         dist_transform1 /= max1 > 0 ? max1 : 1;
//         dist_transform2 /= max2 > 0 ? max2 : 1;

//         // Extended distance transform results in three channels
//         cv::Mat dist_transform1_3c, dist_transform2_3c;
//         cv::merge(std::vector<cv::Mat>{dist_transform1, dist_transform1, dist_transform1}, dist_transform1_3c);
//         cv::merge(std::vector<cv::Mat>{dist_transform2, dist_transform2, dist_transform2}, dist_transform2_3c);

//         cv::Mat blend_weights = dist_transform1_3c + dist_transform2_3c;
//         blend_weights.setTo(1, blend_weights == 0);  // Avoid division by 0

//         // Calculate the blended image using weighted average
//         out_img = (dist_transform1_3c.mul(img1) + dist_transform2_3c.mul(img2)) / blend_weights;
//     }

//     return out_img; 
// }