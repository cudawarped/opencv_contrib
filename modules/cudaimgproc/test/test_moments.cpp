/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

namespace opencv_test { namespace {

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Moments

PARAM_TEST_CASE(Moments, cv::cuda::DeviceInfo, cv::Size, bool, float, int, int, bool)
{
    DeviceInfo devInfo;
    Size size;
    bool isBinary;
    float pcWidth;
    int momentsType;
    int imgType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        isBinary = GET_PARAM(2);
        pcWidth = GET_PARAM(3);
        momentsType = GET_PARAM(4);
        imgType = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::cuda::setDevice(devInfo.deviceID());
    }

    static void drawCircle(cv::Mat& dst, const cv::Vec3f& circle, bool fill)
    {
        dst.setTo(Scalar::all(0));
        cv::circle(dst, Point2f(circle[0], circle[1]), (int)circle[2], Scalar::all(255), fill ? -1 : 1, cv::LINE_AA);
    }
};

vector<double> momentsToVec(cv::Moments moments) {
    return vector<double>{ moments.m00, moments.m10,moments.m01,moments.m20,moments.m11,moments.m02,
    moments.m30, moments.m21,moments.m12,moments.m03,moments.mu20,moments.mu11,moments.mu02,
    moments.mu30, moments.mu21,moments.mu12,moments.mu03 };
}

// have a test for CPU and GPU version should be in the same bit
CUDA_TEST_P(Moments, Accuracy)
{
    Mat imgHost(size, imgType);
    const Rect roi = useRoi ? Rect(1, 0, imgHost.cols - 2, imgHost.rows) : Rect(0, 0, imgHost.cols, imgHost.rows);
    const Vec3f circle(size.width / 2, size.height / 2, size.width * pcWidth);
    drawCircle(imgHost, circle, true);
    const GpuMat imgDevice(imgHost);
    //setBufferPoolUsage(true);
    //setBufferPoolConfig(getDevice(), 10 * (momentsType == CV_64F) ? 64 : 32, 1);
    const cv::Moments moments = cuda::moments(imgDevice(roi), isBinary);
    Mat imgHostFloat; imgHost(roi).convertTo(imgHostFloat, CV_32F);
    const cv::Moments momentsGs = cv::moments(imgHostFloat, isBinary);

    if (momentsType == CV_64F) {
        ASSERT_EQ(momentsGs.m00, moments.m00);
        ASSERT_EQ(momentsGs.m10, moments.m10);
        ASSERT_EQ(momentsGs.m01, moments.m01);
        ASSERT_EQ(momentsGs.m20, moments.m20);
        ASSERT_EQ(momentsGs.m11, moments.m11);
        ASSERT_EQ(momentsGs.m02, moments.m02);
        ASSERT_EQ(momentsGs.m30, moments.m30);
        ASSERT_EQ(momentsGs.m21, moments.m21);
        ASSERT_EQ(momentsGs.m12, moments.m12);
        ASSERT_EQ(momentsGs.m03, moments.m03);
    }
    else {
        //ASSERT_EQ(momentsGs.m00, moments.m00);
        //ASSERT_EQ(momentsGs.m10, moments.m10);
        //ASSERT_EQ(momentsGs.m01, moments.m01);
        //ASSERT_EQ(momentsGs.m20, moments.m20);
        //ASSERT_EQ(momentsGs.m11, moments.m11);
        //ASSERT_EQ(momentsGs.m02, moments.m02);
        //ASSERT_EQ(momentsGs.m30, moments.m30);
        //ASSERT_EQ(momentsGs.m21, moments.m21);
        //ASSERT_EQ(momentsGs.m12, moments.m12);
        //ASSERT_EQ(momentsGs.m03, moments.m03);
        ASSERT_NEAR(momentsGs.m00, moments.m00, 1e0);
        ASSERT_NEAR(momentsGs.m10, moments.m10, 1e0);
        ASSERT_NEAR(momentsGs.m01, moments.m01, 1e0);
        ASSERT_NEAR(momentsGs.m20, moments.m20, 1e0);
        ASSERT_NEAR(momentsGs.m11, moments.m11, 1e0);
        ASSERT_NEAR(momentsGs.m02, moments.m02, 1e0);
        ASSERT_NEAR(momentsGs.m30, moments.m30, 1e0);
        ASSERT_NEAR(momentsGs.m21, moments.m21, 1e0);
        ASSERT_NEAR(momentsGs.m12, moments.m12, 1e0);
        ASSERT_NEAR(momentsGs.m03, moments.m03, 1e0);
    }


    // only for debug use asserts when done
    vector<double> momentsCpuVec = momentsToVec(momentsGs);
    vector<double> momentsGpuVec = momentsToVec(moments);
    double maxRelError = 0;
    for (int i = 0; i < 10/*momentsCpuVec.size()*/; i++) {
        const double err = abs((momentsCpuVec.at(i) - momentsGpuVec.at(i)) / momentsCpuVec.at(i));
        if (err)
            printf("%i: %f, %f, %f\n", i, err, momentsCpuVec.at(i), momentsGpuVec.at(i));
        maxRelError = max(maxRelError, err);
    }
    if (maxRelError != 0)
        printf("  cumulative error %f\n", maxRelError);
}

CUDA_TEST_P(Moments, Async)
{
    Stream stream;
    GpuMat momentsDevice(1, 10, momentsType);
    Mat imgHost(size, imgType);
    const Rect roi = useRoi ? Rect(1, 0, imgHost.cols - 2, imgHost.rows) : Rect(0, 0, imgHost.cols, imgHost.rows);
    const Vec3f circle(size.width / 2, size.height / 2, size.width * pcWidth);
    drawCircle(imgHost, circle, true);
    const GpuMat imgDevice(imgHost);
    cuda::moments(imgDevice(roi), momentsDevice, isBinary, false, stream);
    Mat momentsHost; momentsDevice.download(momentsHost, stream);
    stream.waitForCompletion();
    Mat momentsHost64F = momentsHost;
    if (momentsType == CV_32F)
        momentsHost.convertTo(momentsHost64F, CV_64F);
    const cv::Moments moments = cv::Moments(momentsHost64F.at<double>(0), momentsHost64F.at<double>(1), momentsHost64F.at<double>(2), momentsHost64F.at<double>(3), momentsHost64F.at<double>(4), momentsHost64F.at<double>(5), momentsHost64F.at<double>(6), momentsHost64F.at<double>(7), momentsHost64F.at<double>(8), momentsHost64F.at<double>(9));

    Mat imgHostAdjustedType = imgHost(roi);
    if (imgType != CV_8U && imgType != CV_32F)
        imgHost(roi).convertTo(imgHostAdjustedType, CV_32F);
    auto t1 = std::chrono::high_resolution_clock::now();
    const cv::Moments momentsGs = cv::moments(imgHostAdjustedType, isBinary);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed_time_cpu = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("CPU timer: %4ldus\n", elapsed_time_cpu);

    ASSERT_EQ(momentsGs.m00, moments.m00);
    ASSERT_EQ(momentsGs.m10, moments.m10);
    ASSERT_EQ(momentsGs.m01, moments.m01);
    ASSERT_EQ(momentsGs.m20, moments.m20);
    ASSERT_EQ(momentsGs.m11, moments.m11);
    ASSERT_EQ(momentsGs.m02, moments.m02);
    ASSERT_EQ(momentsGs.m30, moments.m30);
    ASSERT_EQ(momentsGs.m21, moments.m21);
    ASSERT_EQ(momentsGs.m12, moments.m12);
    ASSERT_EQ(momentsGs.m03, moments.m03);


    // only for debug use asserts when done
    vector<double> momentsCpuVec = momentsToVec(momentsGs);
    vector<double> momentsGpuVec = momentsToVec(moments);
    double maxRelError = 0;
    for (int i = 0; i < 10/*momentsCpuVec.size()*/; i++) {
        const double err = abs((momentsCpuVec.at(i) - momentsGpuVec.at(i)) / momentsCpuVec.at(i));
        if (err)
            printf("%i: %f, %f, %f\n", i, err, momentsCpuVec.at(i), momentsGpuVec.at(i));
        maxRelError = max(maxRelError, err);
    }
    if (maxRelError != 0)
        printf("  cumulative error %f\n", maxRelError);

    //{ moments_cpu.m00, moments_cpu.m10, moments_cpu.m01, moments_cpu.m20, moments_cpu.m11, moments_cpu.m02,
    //    moments_cpu.m30, moments_cpu.m21,moments_cpu.m12,moments_cpu.m03,moments_cpu.mu20,moments_cpu.mu11,moments_cpu.mu02,
    //    moments_cpu.mu30, moments_cpu.mu21,moments_cpu.mu12,moments_cpu.mu03 };
    const bool debug = true;


}

//#define SIZES testing::Values(Size(640,480), Size(1280,720), Size(1920,1080))
//#define GRAYSCALE_BINARY testing::Values(true, false)
//#define SHAPE_PC testing::Values(0.1,0.9)
//#define TYPE testing::Values(CV_64F, CV_32F)
//#define USE_ROI testing::Values(true, false)
//#define MIXED_PRECISION testing::Values(true, false)
//#define DEFAULT_STREAM testing::Values(true, false)

//#define SIZES testing::Values(Size(1920,1080))
//#define GRAYSCALE_BINARY testing::Bool()
//#define SHAPE_PC testing::Values(0.9)
//#define MOMENTS_TYPE testing::Values(CV_64F)
//#define USE_ROI testing::Bool()
//#define MIXED_PRECISION testing::Bool()
//#define DEFAULT_STREAM testing::Bool()

#define SIZES DIFFERENT_SIZES
#define GRAYSCALE_BINARY testing::Bool()
#define SHAPE_PC testing::Values(0.1, 0.9)
#define MOMENTS_TYPE testing::Values(CV_32F, CV_64F)
#define IMG_TYPE ALL_DEPTH
#define USE_ROI testing::Bool()

//#define SIZES testing::Values(Size(640,480),Size(1280,720),Size(1920,1080))
//#define GRAYSCALE_BINARY testing::Bool()
//#define SHAPE_PC testing::Values(0.1, 0.9)
//#define MOMENTS_TYPE testing::Values(CV_32F, CV_64F)
//#define IMG_TYPE testing::Values(CV_8U)
//#define USE_ROI testing::Values(false)



//INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, Moments, testing::Combine(ALL_DEVICES, SIZES, GRAYSCALE_BINARY, SHAPE_PC, TYPE, USE_ROI, MIXED_PRECISION, DEFAULT_STREAM));
INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, Moments, testing::Combine(ALL_DEVICES, SIZES, GRAYSCALE_BINARY, SHAPE_PC, MOMENTS_TYPE, IMG_TYPE, USE_ROI));
}} // namespace


#endif // HAVE_CUDA
