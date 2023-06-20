// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

cv::Moments cv::cuda::moments(InputArray _src, bool binary) { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device { namespace imgproc {
        void Moments(const cv::cuda::GpuMat& img, bool binaryImage);

        template <typename T>
        void moments(const PtrStepSzb src, PtrStep<T> moments, const bool binary, const bool mixedPrecision, const int offsetX, const cudaStream_t stream);
}}}}


//cv::Moments cv::cuda::moments(InputArray _src, bool binary) {
//    const cv::cuda::GpuMat src = _src.getGpuMat();
//    int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
//
//    CV_Assert(cn == 1 && depth == CV_8U);
//
//    return cv::cuda::device::imgproc::Moments(src, binary);
//}


// add commnet here to ref other method on what to use
void cv::cuda::createGpuMoments(GpuMat& moments, const int depth) {
    constexpr int nGpuMoments = 10;
    //const int type = useDouble ? CV_64F : CV_32F;
    // could use create func?
    moments = GpuMat(1, nGpuMoments, depth);
    moments.setTo(0);
}

// include Mat


bool cv::cuda::moments(InputArray src_, OutputArray moments_, const bool binary, const bool mixedPrecision, const Stream& stream) {
    const cv::cuda::GpuMat src = src_.getGpuMat();

    // create moments if doesn't exist - which type
    // if use double output array can still be a float
    GpuMat moments = moments_.getGpuMat();
    int momentsDepth = CV_MAT_DEPTH(moments.type());

    //depth = CV_MAT_DEPTH(type)
    if (moments.empty())
        createGpuMoments(moments, momentsDepth);

    //if(moments.empty())
    //    moments.create(1, width, CV_8UC3);

    int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(cn == 1 && depth == CV_8U);

    Point ofs; Size wholeSize;
    src.locateROI(wholeSize, ofs);
    // where to determine if using double?

    if(momentsDepth == CV_32F)
        cv::cuda::device::imgproc::moments<float>(src, moments, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
    else
        cv::cuda::device::imgproc::moments<double>(src, moments, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));

    return true;
}

#endif /* !defined (HAVE_CUDA) */
