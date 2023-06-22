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

        template <typename TSrc, typename TMoments>
        void moments(const PtrStepSz<TSrc> src, PtrStep<TMoments> moments, const bool binary, const bool mixedPrecision, const int offsetX, const cudaStream_t stream);
}}}}


//cv::Moments cv::cuda::moments(InputArray _src, bool binary) {
//    const cv::cuda::GpuMat src = _src.getGpuMat();
//    int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
//
//    CV_Assert(cn == 1 && depth == CV_8U);
//
//    return cv::cuda::device::imgproc::Moments(src, binary);
//}

constexpr int nGpuMoments = 10;
// add commnet here to ref other method on what to use
void cv::cuda::createGpuMoments(GpuMat& moments, const int depth) {
    //const int type = useDouble ? CV_64F : CV_32F;
    // could use create func?
    moments = GpuMat(1, nGpuMoments, depth);
    moments.setTo(0);
}

// include Mat

// overload for Mat maybe

bool cv::cuda::moments(InputArray src_, OutputArray moments_, const bool binary, const bool mixedPrecision, const Stream& stream) {
    // todo
    // make it work with host mem buffer pool etc with comment
    //

    const cv::cuda::GpuMat src = src_.getGpuMat();

    // create moments if doesn't exist - which type
    // if use double output array can still be a float
    GpuMat moments = moments_.getGpuMat();
    //int momentsDepth = CV_MAT_DEPTH(moments.type());

    //depth = CV_MAT_DEPTH(type)
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_32FC1);
    if (moments.empty())
        createGpuMoments(moments, CV_64F);
//    else
    CV_Assert(((moments.type() == CV_32FC1 || moments.type() == CV_64FC1) && moments.rows == 1 && moments.cols >= nGpuMoments));

    //if(moments.empty())
    //    moments.create(1, width, CV_8UC3);

    //int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(src.channels() == 1 && (src.depth() == CV_8U || src.depth() == CV_32F));

    Point ofs; Size wholeSize;
    src.locateROI(wholeSize, ofs);
    // where to determine if using double?

    // why not for all types? - q for cv3d?
    if (src.depth() == CV_8U) {
        if (moments.depth() == CV_32F)
            cv::cuda::device::imgproc::moments<uchar, float>(src, moments, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
        else
            cv::cuda::device::imgproc::moments<uchar, double>(src, moments, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
    }
    else {
        if (moments.depth() == CV_32F)
            cv::cuda::device::imgproc::moments<float, float>(src, moments, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
        else
            cv::cuda::device::imgproc::moments<float, double>(src, moments, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
    }
    //if (momentsDepth == CV_32F)
    //    cv::cuda::device::imgproc::moments<float>(src, moments, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
    //else
    //    cv::cuda::device::imgproc::moments<double>(src, moments, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));

    return true;
}

// how to do this, would need to alocate GpuMoments?
// overload -> would need
// better to dload yourself then calculate the moments? - not going to work well in python?
//cv::Moments cv::cuda::hostMoments()
//// just use bufferbool will be slower than gpu method but
//cv::Moments cv::cuda::moments(InputArray src, GpuMat moments, const bool binary, const bool mixedPrecision, const Stream& stream) {
//    // overload for mat
//    cv::cuda::moments(src, OutputArray moments, const bool binary, const bool mixedPrecision, const Stream & stream)
//
//}

#endif /* !defined (HAVE_CUDA) */
