#include "blendImagePair.h"
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat blendImagePair(const cv::Mat& img1, const cv::Mat& mask1, const cv::Mat& img2, const cv::Mat& mask2, const std::string& mode) {
    // Input: "img1" and "img2" (normalized CV_32F, 3 channel, range 0.0-1.0); "mask1" and "mask2" (normalized CV_32F, one channel 0.0-1.0)
    // Output: "out_img" (CV_32F, 3 channels, range 0.0-0.1)
    cv::Mat out_img;
    cv::Mat mask1_8U, mask2_8U;

    // 将掩码转换为CV_8U并归一化到 0-255 (in python it is 0-1)
    mask1_8U = mask1 > 0;
    mask2_8U = mask2 > 0;
    mask1_8U.convertTo(mask1_8U, CV_8U);
    mask2_8U.convertTo(mask2_8U, CV_8U);

    // // 输出调试信息
    // std::cout << "img1 size: " << img1.size() << ", channels: " << img1.channels() << std::endl;
    // std::cout << "img2 size: " << img2.size() << ", channels: " << img2.channels() << std::endl;
    // std::cout << "mask1 size: " << mask1.size() << ", channels: " << mask1.channels() << std::endl;
    // std::cout << "mask2 size: " << mask2.size() << ", channels: " << mask2.channels() << std::endl;

    if (mode == "overlay") {
        // use mask to overlay img
        out_img = img1.clone();
        img2.copyTo(out_img, mask2_8U);  // 将 img2 复制到 mask2_8u 非零的区域
    } else if (mode == "blend") {
        // 使用距离变换进行平滑混合
        cv::Mat dist_transform1, dist_transform2;

        cv::distanceTransform(mask1_8U, dist_transform1, cv::DIST_L2, 3);
        cv::distanceTransform(mask2_8U, dist_transform2, cv::DIST_L2, 3);

        dist_transform1 /= cv::norm(dist_transform1, cv::NORM_INF);
        dist_transform2 /= cv::norm(dist_transform2, cv::NORM_INF);

        // 扩展距离变换结果为三通道
        cv::Mat dist_transform1_3c, dist_transform2_3c;
        cv::merge(std::vector<cv::Mat>{dist_transform1, dist_transform1, dist_transform1}, dist_transform1_3c);
        cv::merge(std::vector<cv::Mat>{dist_transform2, dist_transform2, dist_transform2}, dist_transform2_3c);

        cv::Mat blend_weights = dist_transform1_3c + dist_transform2_3c;
        blend_weights.setTo(1, blend_weights == 0);  // 避免除以 0

        // 使用加权平均计算混合图像
        out_img = (dist_transform1_3c.mul(img1) + dist_transform2_3c.mul(img2)) / blend_weights;
    }

    return out_img; 
}

/////////////////////////////////////////////////////////////
//测试用代码，编译blendImagePair.cpp之前请注释下面的main() 函数
/////////////////////////////////////////////////////////////
// 将下面主函数复制到外面单独的main函数中，编译： g++ -o blendImagePair main.cpp blendImagePair.cpp `pkg-config --cflags --libs opencv4`
////////////////////////////////////////////////////////////
// 运行  ./blendImagePair
/////////////////////////////////////////////////////////////

// #include <opencv2/opencv.hpp>
// #include "blendImagePair.h"
// #include <iostream>

// int main() {
//     // 加载具有 alpha 通道的图像
//     cv::Mat fish = cv::imread("../photos/blendImagePair/escher_fish.png", cv::IMREAD_UNCHANGED);
//     cv::Mat horse = cv::imread("../photos/blendImagePair/escher_horsemen.png", cv::IMREAD_UNCHANGED);

//     if (fish.empty() || horse.empty()) {
//         std::cerr << "Error: Could not load images." << std::endl;
//         return -1;
//     }

//     // 分离颜色通道和 alpha 通道（作为掩码）
//     std::vector<cv::Mat> fish_channels, horse_channels;
//     cv::split(fish, fish_channels);
//     cv::split(horse, horse_channels);

//     cv::Mat fish_img, horse_img;
//     cv::Mat fish_mask = fish_channels[3];
//     cv::Mat horse_mask = horse_channels[3];
//     cv::merge(std::vector<cv::Mat>{fish_channels[0], fish_channels[1], fish_channels[2]}, fish_img);
//     cv::merge(std::vector<cv::Mat>{horse_channels[0], horse_channels[1], horse_channels[2]}, horse_img);

//     // 归一化图像到 [0, 1] 范围，并转换为浮点型
//     fish_img.convertTo(fish_img, CV_32F, 1.0 / 255.0);
//     horse_img.convertTo(horse_img, CV_32F, 1.0 / 255.0);
//     fish_mask.convertTo(fish_mask, CV_32F, 1.0 / 255.0);
//     horse_mask.convertTo(horse_mask, CV_32F, 1.0 / 255.0);

//     // 测试 blend 模式
//     cv::Mat blended_result = blendImagePair(fish_img, fish_mask, horse_img, horse_mask, "blend");
//     cv::imwrite("../photos/blendImagePair/blended_result.png", blended_result * 255);

//     // 测试 overlay 模式
//     cv::Mat overlay_result = blendImagePair(fish_img, fish_mask, horse_img, horse_mask, "overlay");
//     cv::imwrite("../photos/blendImagePair/overlay_result.png", overlay_result * 255);

//     std::cout << "Images saved successfully in the outputs folder." << std::endl;
//     return 0;
// }