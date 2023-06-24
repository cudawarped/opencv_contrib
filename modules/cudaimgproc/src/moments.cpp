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
        //void Moments(const cv::cuda::GpuMat& img, bool binaryImage);

        template <typename TSrc, typename TMoments>
        void moments(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const bool mixedPrecision, const int offsetX, const cudaStream_t stream);
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

// maybe add the type of moments, why not?
void cv::cuda::moments(InputArray src, OutputArray moments, const bool binary, const bool mixedPrecision, Stream& stream) {
    // todo
    // make it work with host mem buffer pool etc with comment
    //
    const GpuMat srcDevice = getInputMat(src, stream);
    //CV_Assert(src.type() == CV_8UC1 || src.type() == CV_32FC1); // add more types?
   //const cv::cuda::GpuMat src = src_.getGpuMat();

    // create moments if doesn't exist - which type
    // if use double output array can still be a float

    const int momentsType = (moments.type() != CV_32F && moments.type() != CV_64F) ? CV_64F : moments.type();
    GpuMat momentsDevice = getOutputMat(moments, 1, 10, momentsType, stream);

    //GpuMat moments = moments_.getGpuMat();
    //int momentsDepth = CV_MAT_DEPTH(moments.type());

    //depth = CV_MAT_DEPTH(type)

    //if (moments.empty())
    //    createGpuMoments(moments, CV_64F);
//    else
    //CV_Assert(((moments.type() == CV_32FC1 || moments.type() == CV_64FC1) && moments.rows == 1 && moments.cols >= nGpuMoments));

    //if(moments.empty())
    //    moments.create(1, width, CV_8UC3);

    //int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    //CV_Assert(srcDevice.channels() == 1 && (srcDevice.depth() == CV_8U || srcDevice.depth() == CV_32F));

    Point ofs; Size wholeSize;
    srcDevice.locateROI(wholeSize, ofs);
    // where to determine if using double?

    typedef void (*func_t)(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const bool mixedPrecision, const int offsetX, const cudaStream_t stream);

    static const func_t funcs[7][2] =
    {
        {device::imgproc::moments<uchar, float>,  device::imgproc::moments<uchar, double> },
        {device::imgproc::moments<schar, float>,  device::imgproc::moments<schar, double> },
        {device::imgproc::moments<ushort, float>, device::imgproc::moments<ushort, double>},
        {device::imgproc::moments<short, float>,  device::imgproc::moments<short, double> },
        {device::imgproc::moments<int, float>,    device::imgproc::moments<int, double> },
        {device::imgproc::moments<float, float>,  device::imgproc::moments<float, double> },
        {device::imgproc::moments<double, float>, device::imgproc::moments<double, double> }


        //{0 /*warpAffine_gpu<schar>*/, 0 /*warpAffine_gpu<char2>*/  , 0 /*warpAffine_gpu<char3>*/, 0 /*warpAffine_gpu<char4>*/},
        //{warpAffine_gpu<ushort>     , 0 /*warpAffine_gpu<ushort2>*/, warpAffine_gpu<ushort3>    , warpAffine_gpu<ushort4>    },
        //{warpAffine_gpu<short>      , 0 /*warpAffine_gpu<short2>*/ , warpAffine_gpu<short3>     , warpAffine_gpu<short4>     },
        //{0 /*warpAffine_gpu<int>*/  , 0 /*warpAffine_gpu<int2>*/   , 0 /*warpAffine_gpu<int3>*/ , 0 /*warpAffine_gpu<int4>*/ },
        //{warpAffine_gpu<float>      , 0 /*warpAffine_gpu<float2>*/ , warpAffine_gpu<float3>     , warpAffine_gpu<float4>     }
    };

    int i = srcDevice.depth();
    const func_t func = funcs[i][momentsType == CV_64F];
    //CV_Assert(func != 0);

    func(srcDevice, momentsDevice, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
    // why not for all types? - q for cv3d?
    //if (srcDevice.depth() == CV_8U) {
    //    if (momentsDevice.depth() == CV_32F)
    //        cv::cuda::device::imgproc::moments<uchar, float>(srcDevice, momentsDevice, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
    //    else
    //        cv::cuda::device::imgproc::moments<uchar, double>(srcDevice, momentsDevice, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
    //}
    //else {
    //    if (moments.depth() == CV_32F)
    //        cv::cuda::device::imgproc::moments<float, float>(srcDevice, momentsDevice, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
    //    else
    //        cv::cuda::device::imgproc::moments<float, double>(srcDevice, momentsDevice, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
    //}
    //if (momentsDepth == CV_32F)
    //    cv::cuda::device::imgproc::moments<float>(src, moments, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
    //else
    //    cv::cuda::device::imgproc::moments<double>(src, moments, binary, mixedPrecision, ofs.x, StreamAccessor::getStream(stream));
    syncOutput(momentsDevice, moments, stream);
    //return true;
}

Moments cv::cuda::moments(InputArray src, const bool binary, const int momentsType) {
    // add precision here - how to assess
    //constexpr int momentsType = CV_64F;
    HostMem dst;
    moments(src, dst, binary);
    Mat moments = dst.createMatHeader();
    //if(moments.type() != CV32F )

    return Moments(moments.at<double>(0), moments.at<double>(1), moments.at<double>(2), moments.at<double>(3), moments.at<double>(4), moments.at<double>(5), moments.at<double>(6), moments.at<double>(7), moments.at<double>(8), moments.at<double>(9));
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
