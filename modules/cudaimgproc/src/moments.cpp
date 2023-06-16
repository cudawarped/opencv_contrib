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
        Moments Moments(const cv::cuda::GpuMat& img, bool binaryImage);

        template <typename T>
        void Moments1(const PtrStepSzb src, PtrStep<T> moments, bool binary, bool computeCentral, cudaStream_t stream);
}}}}


cv::Moments cv::cuda::moments(InputArray _src, bool binary) {
    const cv::cuda::GpuMat src = _src.getGpuMat();
    int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    CV_Assert(cn == 1 && depth == CV_8U);

    return cv::cuda::device::imgproc::Moments(src, binary);
}


bool cv::cuda::moments1(InputArray src_, OutputArray moments_, const bool binary, const MomentType momentType, const bool useDouble, Stream& stream) {
    const cv::cuda::GpuMat src = src_.getGpuMat();

    // create moments if doesn't exist - which type
    // if use double output array can still be a float
    const GpuMat moments = moments_.getGpuMat();
    int momentsDepth = CV_MAT_DEPTH(moments.type());

    int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    CV_Assert(cn == 1 && depth == CV_8U);

    // where to determine if using double?

    if(momentsDepth == CV_32F)
        cv::cuda::device::imgproc::Moments1<float>(src, moments, binary, momentType == MomentType::CENTRAL, StreamAccessor::getStream(stream));
    else
        cv::cuda::device::imgproc::Moments1<double>(src, moments, binary, momentType == MomentType::CENTRAL, StreamAccessor::getStream(stream));

    return true;
}

#endif /* !defined (HAVE_CUDA) */
