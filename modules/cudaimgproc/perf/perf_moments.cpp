// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {
    namespace {
        static void drawCircle(cv::Mat& dst, const cv::Vec3f& circle, bool fill)
        {
            dst.setTo(cv::Scalar::all(0));
            cv::circle(dst, cv::Point2f(circle[0], circle[1]), (int)circle[2],
                cv::Scalar::all(255), fill ? -1 : 1, cv::LINE_AA);
        }
        DEF_PARAM_TEST(aa, string, int, bool);

        PERF_TEST_P(aa, aaa,
            Combine(Values("perf/800x600.png", "perf/1280x1024.png", "perf/1680x1050.png"),
                Values(3, 5),
                Bool()))
        {


            const cv::Size size = Size(1920, 1920);// GET_PARAM(1);
            const bool isBinary = true;// GET_PARAM(2);
            const float pcWidth = 0.9;

            cv::Vec3f circle(size.width / 2, size.height / 2, size.width * pcWidth);
            Mat imgHost(size, CV_8UC1);
            drawCircle(imgHost, circle, true);
            if (PERF_RUN_CUDA()) {
                // could have createGpuMoments() method with a type
                GpuMat moments(1, 10, CV_32F);
                GpuMat imgDevice(imgHost);
                TEST_CYCLE() cv::cuda::moments1(imgDevice, moments, isBinary, MomentType::SPATIAL, false);
                CUDA_SANITY_CHECK(moments);
            }
            else {
                cv::Moments moments;
                TEST_CYCLE() moments = cv::moments(imgHost, isBinary);
                //CUDA_SANITY_CHECK(moments);
            }



            //circles[0] = cv::Vec3i(10, 10, 4); //cv::Vec3i(20, 20, 10);
            //circles[0] = cv::Vec3i(size.width / 2, size.height / 2, size.width / 2.1);


   /*         int a = 1;
            const string fileName = GET_PARAM(0);
            const int apperture_size = GET_PARAM(1);
            const bool useL2gradient = GET_PARAM(2);

            const cv::Mat image = readImage(fileName, cv::IMREAD_GRAYSCALE);
            ASSERT_FALSE(image.empty());

            const double low_thresh = 50.0;
            const double high_thresh = 100.0;

            if (PERF_RUN_CUDA())
            {
                const cv::cuda::GpuMat d_image(image);
                cv::cuda::GpuMat dst;

                cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, apperture_size, useL2gradient);

                TEST_CYCLE() canny->detect(d_image, dst);

                CUDA_SANITY_CHECK(dst);
            }
            else
            {
                cv::Mat dst;

                TEST_CYCLE() cv::Canny(image, dst, low_thresh, high_thresh, apperture_size, useL2gradient);

                CPU_SANITY_CHECK(dst);
            }
        }*/
        }


    }
}
