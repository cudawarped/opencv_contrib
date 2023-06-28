// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {
static void drawCircle(cv::Mat& dst, const cv::Vec3i& circle, bool fill)
{
    dst.setTo(Scalar::all(0));
    cv::circle(dst, Point2i(circle[0], circle[1]), circle[2], Scalar::all(255), fill ? -1 : 1, cv::LINE_AA);
}

DEF_PARAM_TEST(Sz_Depth, Size, int);
PERF_TEST_P(Sz_Depth, Moments, Combine(CUDA_TYPICAL_MAT_SIZES, Values(CV_32F, CV_64F)))
{
    const cv::Size size = GET_PARAM(0);
    const int momentsType = GET_PARAM(1);
    Mat imgHost(size, CV_8U);
    const Vec3i circle(size.width / 2, size.height / 2, static_cast<int>(static_cast<float>(size.width / 2) * 0.9));
    drawCircle(imgHost, circle, true);
    if (PERF_RUN_CUDA()) {
        const MomentsOrder order = MomentsOrder::THIRD;
        const int nMoments = numMoments(order);
        GpuMat momentsDevice(1, nMoments, momentsType);
        const GpuMat imgDevice(imgHost);
        TEST_CYCLE() cuda::spatialMoments(imgDevice, momentsDevice, false, order, momentsType);
        CUDA_SANITY_CHECK(momentsDevice);
    }
    else {
        cv::Moments momentsHost;
        TEST_CYCLE() momentsHost = cv::moments(imgHost, false);
        SANITY_CHECK_MOMENTS(momentsHost, 2e-4, ERROR_RELATIVE);
    }
}
}}
