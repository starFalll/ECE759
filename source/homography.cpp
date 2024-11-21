#include <iostream>
#include <omp.h>
#include "homography.h"
#include "common.h"

// Function to compute homography matrix
Eigen::Matrix3d computeHomography(const std::vector<Eigen::Vector2d>& src_pts, const std::vector<Eigen::Vector2d>& dest_pts) {
    int n = src_pts.size();
    Eigen::MatrixXd A(2 * n, 9);

    #pragma omp parallel for 
    for (int i = 0; i < n; ++i) {
        double x1 = src_pts[i][0], y1 = src_pts[i][1];
        double x2 = dest_pts[i][0], y2 = dest_pts[i][1];
        
        A.row(2 * i) << x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2;
        A.row(2 * i + 1) << 0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2;
    }

    // Compute eigenvalues and eigenvectors of A^T * A
    Eigen::MatrixXd AtA = A.transpose() * A;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(AtA);
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();

    // Eigenvector with smallest eigenvalue
    Eigen::VectorXd h = eigenvectors.col(0);
    Eigen::Matrix3d H = Eigen::Map<Eigen::Matrix3d>(h.data());

    return H.transpose();
}

// Function to apply homography matrix to source points
std::vector<Eigen::Vector2d> applyHomography(const Eigen::Matrix3d& H, const std::vector<Eigen::Vector2d>& src_pts) {
    int n = src_pts.size();
    std::vector<Eigen::Vector2d> dest_pts(n);
    #pragma omp parallel for 
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d pt(src_pts[i][0], src_pts[i][1], 1.0);
        Eigen::Vector3d transformed_pt = H * pt;
        dest_pts[i] = Eigen::Vector2d(transformed_pt[0] / transformed_pt[2], transformed_pt[1] / transformed_pt[2]);
    }

    return dest_pts;
}

// Function to show correspondences between two images
cv::Mat showCorrespondence(const cv::Mat& img1, const cv::Mat& img2, const std::vector<Eigen::Vector2d>& pts1, const std::vector<Eigen::Vector2d>& pts2) {
    int width = img1.cols + img2.cols;
    int height = std::max(img1.rows, img2.rows);
    cv::Mat result(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Paste img1 and img2 side-by-side
    img1.copyTo(result(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(result(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    // Draw lines for correspondences
    #pragma omp parallel for 
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Point pt1(pts1[i][0], pts1[i][1]);
        cv::Point pt2(pts2[i][0] + img1.cols, pts2[i][1]);
        #pragma omp critical(LOCK_RESULT)
        cv::line(result, pt1, pt2, cv::Scalar(255, 0, 0), 2);
    }

    return result;
}

// int main() {
//     // Load example images
//     cv::Mat orig_img = cv::imread("../photos/homography/portrait.png");
//     cv::Mat warped_img = cv::imread("../photos/homography/portrait_transformed.png");

//     if (orig_img.empty() || warped_img.empty()) {
//         std::cerr << "Could not load images." << std::endl;
//         return -1;
//     }

//     // Selected points
//     std::vector<Eigen::Vector2d> src_pts = { {158, 101}, {332, 316}, {386, 574}, {464, 514}};
//     std::vector<Eigen::Vector2d> dest_pts = { {136, 143}, {261, 287}, {305, 527}, {389, 481} };

//     // Compute homography
//     Eigen::Matrix3d H = computeHomography(src_pts, dest_pts);
//     std::cout<< "homography:"<<H<<std::endl;

//     // Choose another set of points on orig_img for testing.
//     std::vector<Eigen::Vector2d> test_pts = { {158, 101}, {332, 316}, {386, 574}, {464, 514} };

//     // Apply homography
//     std::vector<Eigen::Vector2d> transformed_pts = applyHomography(H, test_pts);
//     for (const auto& point : transformed_pts) {
//         std::cout << "(" << point.x() << ", " << point.y() << ")\n";
//     }

//     // Show correspondences
//     cv::Mat result = showCorrespondence(orig_img, warped_img, test_pts, transformed_pts);

//     // Save the result
//     cv::imwrite("../photos/homography/homography.jpg", result);

//     return 0;
// }
