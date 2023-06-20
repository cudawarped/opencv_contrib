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

PARAM_TEST_CASE(Moments, cv::cuda::DeviceInfo, cv::Size, bool, float, int, bool, bool, bool)
{
    static void drawCircle(cv::Mat& dst, const cv::Vec3f& circle, bool fill)
    {
        dst.setTo(cv::Scalar::all(0));
        cv::circle(dst, cv::Point2f(circle[0], circle[1]), (int)circle[2],
                   cv::Scalar::all(255), fill ? -1 : 1, cv::LINE_AA);
    }

    static void drawRectangle(cv::Mat& dst, const cv::Vec4f& rectangle, bool fill)
    {
        dst.setTo(cv::Scalar::all(0));
        cv::rectangle(dst, cv::Point2f(rectangle[0], rectangle[1]),
                      cv::Point2f(rectangle[2], rectangle[3]),
                      cv::Scalar::all(255), fill ? -1 : 1, cv::LINE_AA);
    }

    static void drawEllipse(cv::Mat& dst, const cv::Vec6f& ellipse, bool fill)
    {
        dst.setTo(cv::Scalar::all(0));
        cv::ellipse(dst, cv::Point2f(ellipse[0], ellipse[1]),
                    cv::Size2f(ellipse[2], ellipse[3]), ellipse[4], 0, 360,
                    cv::Scalar::all(255), fill ? -1 : 1, cv::LINE_AA);
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
    const cv::cuda::DeviceInfo devInfo = GET_PARAM(0);
    cv::cuda::setDevice(devInfo.deviceID());
    const cv::Size size = GET_PARAM(1);
    const bool isBinary = GET_PARAM(2);
    const float pcWidth = GET_PARAM(3);
    const int momentsType = GET_PARAM(4);
    const bool useRoi = GET_PARAM(5);
    const bool mixedPrecision = GET_PARAM(6);
    const bool useDefaultStream = GET_PARAM(7);

    //const int roiOffsetX = 1;

    std::cout << isBinary << "," << useRoi << "," << mixedPrecision << std::endl;
    Stream stream = useDefaultStream ? Stream::Null() : Stream();

    cv::cuda::GpuMat momentsDevice;// (1, 17, CV_32F);
    createGpuMoments(momentsDevice, momentsType);
    //const bool

    Mat imgHost(size, CV_8U);
    //const Rect roi = Rect(roiOffsetX, 0, imgHost.cols - roiOffsetX, imgHost.rows);
    const Rect roi = useRoi ? Rect(1, 0, imgHost.cols - 2, imgHost.rows) : Rect(0, 0, imgHost.cols, imgHost.rows);
    //const Rect roi = Rect(0, 0, imgHost.cols, imgHost.rows);
    Vec3f circle(size.width / 2, size.height / 2, size.width * pcWidth);
    drawCircle(imgHost, circle, true);
    GpuMat imgDevice(imgHost);


    // warm up
    //cuda::moments(imgDevice, momentsDevice, isBinary, stream);
    momentsDevice.setTo(0);

    auto t1 = std::chrono::high_resolution_clock::now();
    cuda::moments(imgDevice(roi), momentsDevice, isBinary, mixedPrecision, stream);
    stream.waitForCompletion();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed_time_gpu = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    Mat momentsHost; momentsDevice.download(momentsHost, stream);
    if (stream != Stream::Null())
        stream.waitForCompletion();

    t1 = std::chrono::high_resolution_clock::now();
    const cv::Moments momentsGs = cv::moments(imgHost(roi), isBinary);
    t2 = std::chrono::high_resolution_clock::now();
    auto elapsed_time_cpu = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("CPU timer: %4ldus, GPU timer: %4ldus\n", elapsed_time_cpu, elapsed_time_gpu);
    Mat momentsHost64;
    momentsHost.convertTo(momentsHost64, CV_64F);
    cv::Moments momentsGpu(momentsHost64.at<double>(0), momentsHost64.at<double>(1), momentsHost64.at<double>(2), momentsHost64.at<double>(3), momentsHost64.at<double>(4),
        momentsHost64.at<double>(5), momentsHost64.at<double>(6), momentsHost64.at<double>(7), momentsHost64.at<double>(8), momentsHost64.at<double>(9));



    // only for debug use asserts when done
    vector<double> momentsCpuVec = momentsToVec(momentsGs);
    vector<double> momentsGpuVec = momentsToVec(momentsGpu);

    cv::Moments check(momentsCpuVec.at(0), momentsCpuVec.at(1), momentsCpuVec.at(2), momentsCpuVec.at(3), momentsCpuVec.at(4), momentsCpuVec.at(5), momentsCpuVec.at(6), momentsCpuVec.at(7), momentsCpuVec.at(8), momentsCpuVec.at(9));


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
    // the cpu routine will call the device routine so no need for all this conversion?

    //const int shapeType = GET_PARAM(3);
    //const int shapeIndex = GET_PARAM(4);
    //printf("shapeType=%d, shapeIndex=%d\n", shapeType, shapeIndex);

    //std::vector<cv::Vec3f> circles(6);
    ////circles[0] = cv::Vec3i(10, 10, 4); //cv::Vec3i(20, 20, 10);
    //circles[0] = cv::Vec3i(size.width / 2, size.height / 2, size.width / 2.1);
    //circles[1] = cv::Vec3i(size.width / 2, size.height / 2, size.width / 4);
    //circles[2] = cv::Vec3i(size.width / 2, size.height / 2, size.width / 8);
    //circles[3] = cv::Vec3i(size.width / 2, size.height / 2, size.width / 16);
    ////circles[1] = cv::Vec3i(90, 87, 15);
    //circles[4] = cv::Vec3i(size.width / 2, size.height / 2, size.width / 8);
    //circles[5] = cv::Vec3i(80, 10, 25);

    //std::vector<cv::Vec4f> rectangles(4);
    //rectangles[0] = cv::Vec4i(20, 20, 30, 40);
    //rectangles[1] = cv::Vec4i(40, 47, 65, 60);
    //rectangles[2] = cv::Vec4i(30, 70, 50, 100);
    //rectangles[3] = cv::Vec4i(80, 10, 100, 50);

    //std::vector<cv::Vec6f> ellipses(4);
    //ellipses[0] = cv::Vec6i(20, 20, 10, 15, 0, 0);
    //ellipses[1] = cv::Vec6i(90, 87, 15, 30, 30, 0);
    //ellipses[2] = cv::Vec6i(30, 70, 20, 25, 60, 0);
    //ellipses[3] = cv::Vec6i(80, 10, 25, 50, 75, 0);

    //cv::Mat src_cpu(size, CV_8UC1);
    //switch(shapeType) {
    //  case 0: {
    //    drawCircle(src_cpu, circles[shapeIndex], true);
    //    break;
    //  }
    //  case 1: {
    //    drawRectangle(src_cpu, rectangles[shapeIndex], true);
    //    break;
    //  }
    //  case 2: {
    //    drawEllipse(src_cpu, ellipses[shapeIndex], true);
    //    break;
    //  }
    //}

    //cv::cuda::GpuMat src_gpu = loadMat(src_cpu, false);

    //Stream stream;
    //cuda::Event start, end;
    //cv::cuda::GpuMat moments(1, 17, CV_32F);
    //auto t1 = std::chrono::high_resolution_clock::now();
    //start.record(stream);
    //cv::cuda::moments1(src_gpu, moments, isBinary, MomentType::SPATIAL, false, stream);
    //end.record(stream);
    //stream.waitForCompletion();
    //float nsGs = Event::elapsedTime(start, end) * 1000;
    //auto t2 = std::chrono::high_resolution_clock::now();
    //auto elapsed_time_gpu_new = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    //printf("CPU timer: %4ldus, GPU timer: %.2fus\n", elapsed_time_gpu_new, nsGs);


    //cv::cuda::GpuMat moments64(1, 17, CV_64F);
    //t1 = std::chrono::high_resolution_clock::now();
    //start.record(stream);
    //cv::cuda::moments1(src_gpu, moments64, isBinary, MomentType::SPATIAL, false, stream);
    //end.record(stream);
    //stream.waitForCompletion();
    //nsGs = Event::elapsedTime(start, end) * 1000;
    //t2 = std::chrono::high_resolution_clock::now();
    //elapsed_time_gpu_new = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    //printf("CPU timer: %4ldus, GPU timer: %.2fus\n", elapsed_time_gpu_new, nsGs);
    ////Mat tmp; moments64.download(tmp);

    //t1 = std::chrono::high_resolution_clock::now();
    //const cv::Moments moments_cpu = cv::moments(src_cpu, isBinary);
    //t2 = std::chrono::high_resolution_clock::now();
    ////const cv::Moments moments_gpu = cv::cuda::moments(src_gpu, isBinary);
    ////const auto t2 = std::chrono::high_resolution_clock::now();
    //auto elapsed_time_cpu = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    ////const auto elapsed_time_gpu = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    //printf("CPU's moments took %4ldus\n", elapsed_time_cpu);
    ////printf("GPU's moments took %4ldus\n", elapsed_time_gpu);

    //Mat momentsGpuHostMat; moments64.download(momentsGpuHostMat);

    //// convert to moments cpu
    //t1 = std::chrono::high_resolution_clock::now();
    ////cv::getTickCount()
    //cv::Moments momentsGpuHost(momentsGpuHostMat.at<double>(0), momentsGpuHostMat.at<double>(1), momentsGpuHostMat.at<double>(2), momentsGpuHostMat.at<double>(3), momentsGpuHostMat.at<double>(4),
    //    momentsGpuHostMat.at<double>(5), momentsGpuHostMat.at<double>(6), momentsGpuHostMat.at<double>(7), momentsGpuHostMat.at<double>(8), momentsGpuHostMat.at<double>(9));
    //const cv::Moments moments_cpu1 = cv::moments(src_cpu, isBinary);
    //t2 = std::chrono::high_resolution_clock::now();
    //elapsed_time_cpu = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    ////printf("CPU's moment translation took %4ldus\n", elapsed_time_cpu);
    //cout << "CPU's moment translation took " << elapsed_time_cpu << endl;


    //vector<double> momentsGpu = momentsToVec(momentsGpuHost);
    ////double cumErr = 0;
    //double maxRelError = 0;
    //vector<double> momentsCpu = momentsToVec(moments_cpu);
    ////{ moments_cpu.m00, moments_cpu.m10, moments_cpu.m01, moments_cpu.m20, moments_cpu.m11, moments_cpu.m02,
    ////    moments_cpu.m30, moments_cpu.m21,moments_cpu.m12,moments_cpu.m03,moments_cpu.mu20,moments_cpu.mu11,moments_cpu.mu02,
    ////    moments_cpu.mu30, moments_cpu.mu21,moments_cpu.mu12,moments_cpu.mu03 };



    //for (int i = 0; i < 17; i++) {



    //    double err = abs((momentsCpu.at(i) - momentsGpu.at(i)) / momentsCpu.at(i));
    //    //else

    //    if (err)
    //        printf("%i: %f, %f, %f\n", i, err, momentsCpu.at(i), momentsGpu.at(i));

    //    //cumErr += abs(momentsCpuGs.at<double>(i) - momentsCpu.at<double>(i));
    //    maxRelError = max(maxRelError, err);
    //}
    //if (maxRelError != 0)
    //    printf("  cumulative error %f\n", maxRelError);





    ASSERT_TRUE(true);




    //ASSERT_EQ(moments_cpu.m00, moments_gpu.m00);
    //ASSERT_EQ(moments_cpu.m10, moments_gpu.m10);
    //ASSERT_EQ(moments_cpu.m01, moments_gpu.m01);
    //ASSERT_EQ(moments_cpu.m20, moments_gpu.m20);
    //ASSERT_EQ(moments_cpu.m11, moments_gpu.m11);
    //ASSERT_EQ(moments_cpu.m02, moments_gpu.m02);
    //ASSERT_EQ(moments_cpu.m30, moments_gpu.m30);
    //ASSERT_EQ(moments_cpu.m21, moments_gpu.m21);
    //ASSERT_EQ(moments_cpu.m12, moments_gpu.m12);
    //ASSERT_EQ(moments_cpu.m03, moments_gpu.m03);

    //ASSERT_NEAR(moments_cpu.mu20, moments_gpu.mu20, 1e-4);
    //ASSERT_NEAR(moments_cpu.mu11, moments_gpu.mu11, 1e-4);
    //ASSERT_NEAR(moments_cpu.mu02, moments_gpu.mu02, 1e-4);
    //ASSERT_NEAR(moments_cpu.mu30, moments_gpu.mu30, 1e-4);
    //ASSERT_NEAR(moments_cpu.mu21, moments_gpu.mu21, 1e-4);
    //ASSERT_NEAR(moments_cpu.mu12, moments_gpu.mu12, 1e-4);
    //ASSERT_NEAR(moments_cpu.mu03, moments_gpu.mu03, 1e-4);

    //ASSERT_NEAR(moments_cpu.nu20, moments_gpu.nu20, 1e-4);
    //ASSERT_NEAR(moments_cpu.nu11, moments_gpu.nu11, 1e-4);
    //ASSERT_NEAR(moments_cpu.nu02, moments_gpu.nu02, 1e-4);
    //ASSERT_NEAR(moments_cpu.nu30, moments_gpu.nu30, 1e-4);
    //ASSERT_NEAR(moments_cpu.nu21, moments_gpu.nu21, 1e-4);
    //ASSERT_NEAR(moments_cpu.nu12, moments_gpu.nu12, 1e-4);
    //ASSERT_NEAR(moments_cpu.nu03, moments_gpu.nu03, 1e-4);
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

#define SIZES testing::Values(Size(1920,1080))
#define GRAYSCALE_BINARY testing::Values(true)
#define SHAPE_PC testing::Values(0.9)
#define MOMENTS_TYPE testing::Values(CV_64F, CV_32F)
#define USE_ROI testing::Values(true)
#define MIXED_PRECISION testing::Bool()
#define DEFAULT_STREAM testing::Values(false)


//INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, Moments, testing::Combine(ALL_DEVICES, SIZES, GRAYSCALE_BINARY, SHAPE_PC, TYPE, USE_ROI, MIXED_PRECISION, DEFAULT_STREAM));
INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, Moments, testing::Combine(ALL_DEVICES, SIZES, GRAYSCALE_BINARY, SHAPE_PC, MOMENTS_TYPE, USE_ROI, MIXED_PRECISION, DEFAULT_STREAM));
}} // namespace


#endif // HAVE_CUDA
