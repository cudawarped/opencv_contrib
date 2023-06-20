// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#if !defined CUDA_DISABLER
//#include "device_functions.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/emulation.hpp"
#include "opencv2/core/cuda/transform.hpp"

//#include "opencv2/core/cuda/common.hpp"
//#include "opencv2/core/cuda/emulation.hpp"
//#include "opencv2/core/cuda/transform.hpp"
//#include "opencv2/core/cuda/functional.hpp"
//#include "opencv2/core/cuda/utility.hpp"
//#include "opencv2/core/cuda.hpp"
//#include <opencv2/core/cuda_stream_accessor.hpp>

using namespace cv::cuda;
using namespace cv::cuda::device;


namespace cv { namespace cuda { namespace device { namespace imgproc {

constexpr int blockSizeX = 32;
constexpr int blockSizeY = 16;
//constexpr int momentsSize = sizeof(cv::Moments) / sizeof(double);
//
//constexpr int m00 = offsetof(cv::Moments, m00) / sizeof(double);
//constexpr int m10 = offsetof(cv::Moments, m10) / sizeof(double);
//constexpr int m01 = offsetof(cv::Moments, m01) / sizeof(double);
//constexpr int m20 = offsetof(cv::Moments, m20) / sizeof(double);
//constexpr int m11 = offsetof(cv::Moments, m11) / sizeof(double);
//constexpr int m02 = offsetof(cv::Moments, m02) / sizeof(double);
//constexpr int m30 = offsetof(cv::Moments, m30) / sizeof(double);
//constexpr int m21 = offsetof(cv::Moments, m21) / sizeof(double);
//constexpr int m12 = offsetof(cv::Moments, m12) / sizeof(double);
//constexpr int m03 = offsetof(cv::Moments, m03) / sizeof(double);
//
//constexpr int mu20 = offsetof(cv::Moments, mu20) / sizeof(double);
//constexpr int mu11 = offsetof(cv::Moments, mu11) / sizeof(double);
//constexpr int mu02 = offsetof(cv::Moments, mu02) / sizeof(double);
//constexpr int mu30 = offsetof(cv::Moments, mu30) / sizeof(double);
//constexpr int mu21 = offsetof(cv::Moments, mu21) / sizeof(double);
//constexpr int mu12 = offsetof(cv::Moments, mu12) / sizeof(double);
//constexpr int mu03 = offsetof(cv::Moments, mu03) / sizeof(double);

__global__ void ComputeSpatialMoments(const cuda::PtrStepSzb img, bool binary, double* moments) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < img.rows && x < img.cols) {
      const unsigned int img_index = y * img.step + x;
      const unsigned char val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;
      if (val > 0) {
        const unsigned long x2 = x * x, x3 = x2 * x;
        const unsigned long y2 = y * y, y3 = y2 * y;

        atomicAdd(&moments[0],           val);
        atomicAdd(&moments[1], x       * val);
        atomicAdd(&moments[2],      y  * val);
        atomicAdd(&moments[3], x2      * val);
        atomicAdd(&moments[4], x  * y  * val);
        atomicAdd(&moments[5],      y2 * val);
        atomicAdd(&moments[6], x3      * val);
        atomicAdd(&moments[6], x2 * y  * val);
        atomicAdd(&moments[8], x  * y2 * val);
        atomicAdd(&moments[9],      y3 * val);
      }
    }
}

__global__ void ComputeSpatialMoments(const cuda::PtrStepSzb img, bool binary, float* moments) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < img.rows && x < img.cols) {
        const unsigned int img_index = y * img.step + x;
        const unsigned char val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;
        if (val > 0) {
            const unsigned long x2 = x * x, x3 = x2 * x;
            const unsigned long y2 = y * y, y3 = y2 * y;

            atomicAdd(&moments[0], val);
            atomicAdd(&moments[1], x * val);
            atomicAdd(&moments[2], y * val);
            atomicAdd(&moments[3], x2 * val);
            atomicAdd(&moments[4], x * y * val);
            atomicAdd(&moments[5], y2 * val);
            atomicAdd(&moments[6], x3 * val);
            atomicAdd(&moments[7], x2 * y * val);
            atomicAdd(&moments[8], x * y2 * val);
            atomicAdd(&moments[9], y3 * val);
        }
    }
}



template <typename T>
__global__ void ComputeSpatialMomentsSharedFullReduction(const cuda::PtrStepSzb img, bool binary, T* moments) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ T smem[10][32];

    if (threadIdx.x < 10)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    T val = 0;
    if (y < img.rows && x < img.cols) {
        const unsigned int img_index = y * img.step + x;
        val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;
    }

    const unsigned long x2 = x * x, x3 = x2 * x;
    const unsigned long y2 = y * y, y3 = y2 * y;
    T res = warpButterflyReduce(val);
    if (res) {
        smem[0][threadIdx.y] = res;
        smem[1][threadIdx.y] = warpButterflyReduce(x * val);
        smem[2][threadIdx.y] = y * res;
        smem[3][threadIdx.y] = warpButterflyReduce(x2 * val);
        smem[4][threadIdx.y] = warpButterflyReduce(x * y * val);
        smem[5][threadIdx.y] = y2 * res;
        smem[6][threadIdx.y] = warpButterflyReduce(x3 * val);
        smem[7][threadIdx.y] = warpButterflyReduce(x2 * y * val);
        smem[8][threadIdx.y] = warpButterflyReduce(x * y2 * val);
        smem[9][threadIdx.y] = y3 * res;
    }
    __syncthreads();

    if (threadIdx.x < blockSizeY && threadIdx.y < 10)
        smem[threadIdx.y][0] = halfWarpButterflyReduce(smem[threadIdx.y][threadIdx.x]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < 10)
        atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
}


template <typename T >
__device__ __forceinline__ T warpButterflyReduce(T value) {
    for (int i = 16; i >= 1; i /= 2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
    return value;
}

template <typename T>
__device__ __forceinline__ T halfWarpButterflyReduce(T value) {
    for (int i = 8; i >= 1; i /= 2)
        value += __shfl_xor_sync(0xffff, value, i, 32);
    return value;
}

template <typename T>
__global__ void ComputeSpatialMomentsSharedFullReductionS1(const PtrStepSzb img, const bool binary, T* moments) {
    constexpr int nMoments = 10;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ T smem[blockSizeY][nMoments];

    if (threadIdx.y < nMoments && threadIdx.x < blockSizeY)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    unsigned int val = 0;
    if (y < img.rows && x < img.cols)
        val = (!binary || img(y,x) == 0) ? img(y, x) : 1;

    const unsigned long x2 = x * x, x3 = x2 * x;
    const unsigned long y2 = y * y, y3 = y2 * y;
    T res = warpButterflyReduce(static_cast<T>(val));
    if (res) {
        smem[threadIdx.y][0] = res;
        smem[threadIdx.y][2] = y * res;
        smem[threadIdx.y][5] = y2 * res;
        smem[threadIdx.y][9] = y3 * res;
        smem[threadIdx.y][1] = warpButterflyReduce(x * static_cast<T>(val));
        smem[threadIdx.y][3] = warpButterflyReduce(x2 * static_cast<T>(val));
        smem[threadIdx.y][4] = warpButterflyReduce(x * y * static_cast<T>(val));
        smem[threadIdx.y][6] = warpButterflyReduce(x3 * static_cast<T>(val));
        smem[threadIdx.y][7] = warpButterflyReduce(x2 * y * static_cast<T>(val));
        smem[threadIdx.y][8] = warpButterflyReduce(x * y2 * static_cast<T>(val));
    }
    __syncthreads();

    if (threadIdx.x < blockSizeY && threadIdx.y < nMoments)
        smem[threadIdx.y][0] = halfWarpButterflyReduce(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < nMoments) {
        if(smem[threadIdx.x][0])
            atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
    }
}

// binary could be a template arg aswell
template <typename T>
__global__ void ComputeSpatialMomentsSharedFullReductionS2(const PtrStepSzb img, const bool binary, T* moments) {
    constexpr int nMoments = 10;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ T smem[blockSizeY][nMoments];

    if (threadIdx.y < nMoments && threadIdx.x < blockSizeY)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    unsigned int val = 0;
    if (y < img.rows && x < img.cols)
        val = (!binary || img(y, x) == 0) ? img(y, x) : 1;

    const unsigned long x2 = x * x, x3 = x2 * x;
    const unsigned long y2 = y * y, y3 = y2 * y;
    //T reductions[4] = { 0 };
    T res = warpButterflyReduce(static_cast<T>(val));
    if (res) {
        smem[threadIdx.y][0] = res;
        smem[threadIdx.y][2] = y * res;
        smem[threadIdx.y][5] = y2 * res;
        smem[threadIdx.y][9] = y3 * res;

        //reductions[1] = warpButterflyReduce(x * static_cast<T>(val));
        //reductions[2] = warpButterflyReduce(x2 * static_cast<T>(val));
        //reductions[3] = warpButterflyReduce(x3 * static_cast<T>(val));



        smem[threadIdx.y][1] = warpButterflyReduce(x * static_cast<T>(val));
        smem[threadIdx.y][3] = warpButterflyReduce(x2 * static_cast<T>(val));
        smem[threadIdx.y][6] = warpButterflyReduce(x3 * static_cast<T>(val));

        smem[threadIdx.y][4] = smem[threadIdx.y][1] * y;
        smem[threadIdx.y][7] = smem[threadIdx.y][3] * y;
        smem[threadIdx.y][8] = smem[threadIdx.y][1] * y2;
        //smem[threadIdx.y][4] = warpButterflyReduce(x * y * static_cast<T>(val));
        //smem[threadIdx.y][6] = warpButterflyReduce(x3 * static_cast<T>(val));
        //smem[threadIdx.y][7] = warpButterflyReduce(x2 * y * static_cast<T>(val));
        //smem[threadIdx.y][8] = warpButterflyReduce(x * y2 * static_cast<T>(val));
    }
    __syncthreads();

    if (threadIdx.x < blockSizeY && threadIdx.y < nMoments)
        smem[threadIdx.y][0] = halfWarpButterflyReduce(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < nMoments) {
        if (smem[threadIdx.x][0])
            atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
    }
}

template <typename T>
__global__ void ComputeSpatialMomentsSharedFullReductionS3(const PtrStepSzb img, const bool binary, T* moments) {
    constexpr int nMoments = 10;
    //const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.x * blockDim.y + threadIdx.y;
    __shared__ T smem[blockSizeY][nMoments];

    if (threadIdx.y < nMoments && threadIdx.x < blockSizeY)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    //unsigned int val = 0;
    //if (y < img.rows && x < img.cols)
    //    val = (!binary || img(y, x) == 0) ? img(y, x) : 1;

    T reductions[4] = { 0 };
    if (y < img.rows) {

        //printf("%i\n", y);
        for (int x = threadIdx.x; x < img.cols; x += blockDim.x) {
            const unsigned long x2 = x * x, x3 = x2 * x;
            T val = (!binary || img(y, x) == 0) ? img(y, x) : 1;
            reductions[0] += val;
            //if(img(y,x)> 0)
            //    printf("%i\n", y);
            reductions[1] += val * x;
            reductions[2] += val * x2;
            reductions[3] += val * x3;
            //smem[threadIdx.y][1] = warpButterflyReduce(x * static_cast<T>(val));
            //smem[threadIdx.y][3] = warpButterflyReduce(x2 * static_cast<T>(val));
            //smem[threadIdx.y][6] = warpButterflyReduce(x3 * static_cast<T>(val));

        }
    }

    //const unsigned long x2 = x * x, x3 = x2 * x;
    const unsigned long y2 = y * y, y3 = y2 * y;
    //T reductions[4] = { 0 };
    T res = warpButterflyReduce(static_cast<T>(reductions[0]));
    if (res) {
        //printf("%f\n", res);
        smem[threadIdx.y][0] = res;
        smem[threadIdx.y][2] = y * res;
        smem[threadIdx.y][5] = y2 * res;
        smem[threadIdx.y][9] = y3 * res;

        //reductions[1] = warpButterflyReduce(x * static_cast<T>(val));
        //reductions[2] = warpButterflyReduce(x2 * static_cast<T>(val));
        //reductions[3] = warpButterflyReduce(x3 * static_cast<T>(val));



        smem[threadIdx.y][1] = warpButterflyReduce(reductions[1]);
        smem[threadIdx.y][3] = warpButterflyReduce(reductions[2]);
        smem[threadIdx.y][6] = warpButterflyReduce(reductions[3]);

        smem[threadIdx.y][4] = smem[threadIdx.y][1] * y;
        smem[threadIdx.y][7] = smem[threadIdx.y][3] * y;
        smem[threadIdx.y][8] = smem[threadIdx.y][1] * y2;
        //smem[threadIdx.y][4] = warpButterflyReduce(x * y * static_cast<T>(val));
        //smem[threadIdx.y][6] = warpButterflyReduce(x3 * static_cast<T>(val));
        //smem[threadIdx.y][7] = warpButterflyReduce(x2 * y * static_cast<T>(val));
        //smem[threadIdx.y][8] = warpButterflyReduce(x * y2 * static_cast<T>(val));
    }
    __syncthreads();

    if (threadIdx.x < blockSizeY && threadIdx.y < nMoments)
        smem[threadIdx.y][0] = halfWarpButterflyReduce(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < nMoments) {
        if (smem[threadIdx.x][0])
            atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
    }
}


constexpr int nMoments = 10;

template <typename T>
__device__ __forceinline__ T butterflyWarpReduction(T value) {
    for (int i = 16; i >= 1; i /= 2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
    return value;
}

template <typename T>
__device__ __forceinline__ T butterflyHalfWarpReduction(T value) {
    for (int i = 8; i >= 1; i /= 2)
        value += __shfl_xor_sync(0xffff, value, i, 32);
    return value;
}

template<typename T, typename M>
__device__ void rowReductions(const PtrStepSzb img, const bool binary, const unsigned int y, float& r0, M& r1, M& r2, T& r3, T smem[][nMoments]) {
    for (int x = threadIdx.x; x < img.cols; x += blockDim.x) {
        const unsigned int x2 = x * x;
        const T x3 = static_cast<T>(x2) * x;
        const float val = (!binary || img(y, x) == 0) ? img(y, x) : 1;
        r0 += val;
        r1 += static_cast<M>(val) * x;
        r2 += static_cast<M>(val) * x2;
        //r3 += static_cast<M>(val) * x3;
        r3 += static_cast<T>(val) * x3;
    }
}

template<typename T, bool fourByteAligned, typename M>
__device__ void rowReductionsCoalesced(const PtrStepSzb img, const bool binary, const unsigned int y, float& r0, M& r1, M& r2, T& r3, const int offsetX, T smem[][nMoments]) {
    const int alignedOffset = fourByteAligned ? 0 : 4 - offsetX;
    // load uncoalesced head
    if (!fourByteAligned && threadIdx.x == 0) {
        for (int x = 0; x < ::min(alignedOffset, static_cast<int>(img.cols)); x++) {
            const unsigned int x2 = x * x;
            const M x3 = static_cast<M>(x2) * x;
            const float val = (!binary || img(y, x) == 0) ? img(y, x) : 1;
            r0 += val;
            r1 += static_cast<M>(val) * x;
            r2 += static_cast<M>(val) * x2;
            r3 += static_cast<M>(val) * x3;
        }
    }

    // coalesced loads
    const unsigned int* rowPtrIntAligned = (const unsigned int*)(fourByteAligned ? img.ptr(y) : img.ptr(y) + alignedOffset);
    const int cols4 = fourByteAligned ? img.cols / 4 : (img.cols - alignedOffset) / 4;
    for (int x = threadIdx.x; x < cols4; x += blockDim.x) {
        const unsigned int data = rowPtrIntAligned[x];
#pragma unroll 4
        for (int i = 0; i < 4; i++) {
            const int iX = alignedOffset + 4 * x + i;
            const unsigned int  x2 = iX * iX;
            const M x3 = static_cast<M>(x2) * iX;
            const uchar ucharVal = ((data >> i * 8) & 0xFFU);
            const float val = (!binary || ucharVal == 0) ? ucharVal : 1;
            r0 += val;
            r1 += static_cast<M>(val) * iX;
            r2 += static_cast<M>(val) * x2;
            r3 += static_cast<M>(val) * x3;
        }
    }

    // load uncoalesced tail
    if (threadIdx.x == 0) {
        const int iTailStart = fourByteAligned ? img.cols : cols4 * 4 + alignedOffset;
        for (int x = iTailStart; x < img.cols; x++) {
            const unsigned int x2 = x * x;
            const M x3 = static_cast<M>(x2) * x;
            const float val = (!binary || img(y, x) == 0) ? img(y, x) : 1;
            r0 += val;
            r1 += static_cast<M>(val) * x;
            r2 += static_cast<M>(val) * x2;
            r3 += static_cast<M>(val) * x3;
        }
    }
}

template <typename T, bool coalesced = false, bool fourByteAligned = false, typename M = T>
__global__ void spatialMoments(const PtrStepSzb img, const bool binary, T* moments, const int offsetX = 0) {
    //constexpr int nMoments = 10;
    const unsigned int y = blockIdx.x * blockDim.y + threadIdx.y;
    __shared__ T smem[blockSizeY][nMoments];

    if (threadIdx.y < nMoments && threadIdx.x < blockSizeY)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    float r0 = 0;
    M r1 = 0;
    M r2 = 0;
    T r3 = 0;
    //M reductions[3] = { 0 };
    if (y < img.rows) {
        if (coalesced)
            rowReductionsCoalesced<T, fourByteAligned, M>(img, binary, y, r0, r1, r2, r3, offsetX, smem);
        else
            rowReductions<T, M>(img, binary, y, r0, r1, r2, r3, smem);
    }

    const unsigned long y2 = y * y;
    const T y3 = static_cast<T>(y2) * y;
    float res = butterflyWarpReduction<T>(r0);
    if (res) {
        smem[threadIdx.y][0] = res; //0th
        smem[threadIdx.y][2] = y * static_cast<T>(res); //1st
        smem[threadIdx.y][5] = y2 * static_cast<T>(res); //2nd
        smem[threadIdx.y][9] = y3 * static_cast<T>(res); //3rd
        smem[threadIdx.y][1] = butterflyWarpReduction<T>(r1); //1st
        smem[threadIdx.y][3] = butterflyWarpReduction<T>(r2); //2nd
        smem[threadIdx.y][6] = butterflyWarpReduction<T>(r3); //3rd
        smem[threadIdx.y][4] = smem[threadIdx.y][1] * y; //2nd
        smem[threadIdx.y][7] = smem[threadIdx.y][3] * y; //3rd
        smem[threadIdx.y][8] = smem[threadIdx.y][1] * y2; //3rd
    }
    __syncthreads();

    if (threadIdx.x < blockSizeY && threadIdx.y < nMoments)
        smem[threadIdx.y][0] = butterflyHalfWarpReduction(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < nMoments) {
        if (smem[threadIdx.x][0])
            atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
    }
}





//template<typename T, typename M = T>
//__device__ CalcMoments(const int y, const float val, const M[3] partialReductions, ) {
//    const unsigned long y2 = y * y;
//    const T y3 = static_cast<T>(y2) * y;
//    float res = butterflyWarpReduction<T>(valReduction);
//    if (res) {
//        smem[threadIdx.y][0] = res;
//        smem[threadIdx.y][2] = y * static_cast<T>(res);
//        smem[threadIdx.y][5] = y2 * static_cast<T>(res);
//        smem[threadIdx.y][9] = y3 * static_cast<T>(res);
//        smem[threadIdx.y][1] = butterflyWarpReduction<T>(reductions[0]);
//        smem[threadIdx.y][3] = butterflyWarpReduction<T>(reductions[1]);
//        smem[threadIdx.y][6] = butterflyWarpReduction<T>(reductions[2]);
//        smem[threadIdx.y][4] = smem[threadIdx.y][1] * y;
//        smem[threadIdx.y][7] = smem[threadIdx.y][3] * y;
//        smem[threadIdx.y][8] = smem[threadIdx.y][1] * y2;
//    }
//}

//template<typename T>
//__device__ RowReduce




template <typename T, typename M = T>
__global__ void ComputeSpatialMomentsSharedFullReductionS3a(const PtrStepSzb img, const bool binary, T* moments) {
    constexpr int nMoments = 10;
    const unsigned int y = blockIdx.x * blockDim.y + threadIdx.y;
    __shared__ T smem[blockSizeY][nMoments];

    if (threadIdx.y < nMoments && threadIdx.x < blockSizeY)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    float valReduction = 0;
    M reductions[3] = { 0 };
    if (y < img.rows) {
        for (int x = threadIdx.x; x < img.cols; x += blockDim.x) {
            const unsigned long x2 = x * x;
            const M x3 = static_cast<M>(x2) * x;
            const float val = (!binary || img(y, x) == 0) ? img(y, x) : 1;
            valReduction += val;
            reductions[0] += static_cast<M>(val) * x;
            reductions[1] += static_cast<M>(val) * x2;
            reductions[2] += static_cast<M>(val) * x3;
        }
    }

    const unsigned long y2 = y * y;
    const T y3 = static_cast<T>(y2) * y;
    float res = butterflyWarpReduction<T>(valReduction);
    if (res) {
        smem[threadIdx.y][0] = res;
        smem[threadIdx.y][2] = y * static_cast<T>(res);
        smem[threadIdx.y][5] = y2 * static_cast<T>(res);
        smem[threadIdx.y][9] = y3 * static_cast<T>(res);
        smem[threadIdx.y][1] = butterflyWarpReduction<T>(reductions[0]);
        smem[threadIdx.y][3] = butterflyWarpReduction<T>(reductions[1]);
        smem[threadIdx.y][6] = butterflyWarpReduction<T>(reductions[2]);
        smem[threadIdx.y][4] = smem[threadIdx.y][1] * y;
        smem[threadIdx.y][7] = smem[threadIdx.y][3] * y;
        smem[threadIdx.y][8] = smem[threadIdx.y][1] * y2;
    }
    __syncthreads();

    if (threadIdx.x < blockSizeY && threadIdx.y < nMoments)
        smem[threadIdx.y][0] = butterflyHalfWarpReduction(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < nMoments) {
        if (smem[threadIdx.x][0])
            atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
    }
}

template <bool fourByteAligned, typename T, typename M = T>
__global__ void ComputeSpatialMomentsSharedFullReductionS4(const PtrStepSzb img, const bool binary, T* moments, const int offsetX = 0) {
    constexpr int nMoments = 10;
    const unsigned int y = blockIdx.x * blockDim.y + threadIdx.y;
    int alignedOffset = fourByteAligned ? 0 : 4 - offsetX;
    __shared__ T smem[blockSizeY][nMoments];

    if (threadIdx.y < nMoments && threadIdx.x < blockSizeY)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    float valReduction = 0;
    M reductions[3] = { 0 };
    if (y < img.rows) {
        // load uncoalesced head
        if (!fourByteAligned && threadIdx.x == 0) {
            for (int x = 0; x < ::min(alignedOffset, static_cast<int>(img.cols)); x++) {
                const unsigned int x2 = x * x;
                const M x3 = static_cast<M>(x2) * x;
                const float val = (!binary || img(y, x) == 0) ? img(y, x) : 1;
                valReduction += val;
                reductions[0] += static_cast<M>(val) * x;
                reductions[1] += static_cast<M>(val) * x2;
                reductions[2] += static_cast<M>(val) * x3;
            }
        }

        // coalesced loads
        const unsigned int* rowPtrIntAligned = (const unsigned int*)(fourByteAligned ? img.ptr(y) : img.ptr(y) + alignedOffset);
        const int cols4 = fourByteAligned ? img.cols / 4 : (img.cols - alignedOffset) / 4;
        for (int x = threadIdx.x; x < cols4; x += blockDim.x) {
            const unsigned int data = rowPtrIntAligned[x];
#pragma unroll 4
            for (int i = 0; i < 4; i++) {
                const int iX = alignedOffset + 4 * x + i;
                const unsigned int  x2 = iX * iX;
                const M x3 = static_cast<M>(x2) * iX;
                const uchar ucharVal = ((data >> i * 8) & 0xFFU);
                const float val = (!binary || ucharVal == 0) ? ucharVal : 1;
                valReduction += val;
                reductions[0] += static_cast<M>(val) * iX;
                reductions[1] += static_cast<M>(val) * x2;
                reductions[2] += static_cast<M>(val) * x3;
            }
        }

        // load uncoalesced tail
        if (threadIdx.x == 0) {
            const int iTailStart = fourByteAligned ? img.cols : cols4 * 4 + alignedOffset;
            for (int x = iTailStart; x < img.cols; x++) {
                const unsigned int x2 = x * x;
                const M x3 = static_cast<M>(x2) * x;
                const float val = (!binary || img(y, x) == 0) ? img(y, x) : 1;
                valReduction += val;
                reductions[0] += static_cast<M>(val) * x;
                reductions[1] += static_cast<M>(val) * x2;
                reductions[2] += static_cast<M>(val) * x3;

            }
        }
    }

    const unsigned int y2 = y * y;
    const T y3 = static_cast<T>(y2) * y;
    float res = warpButterflyReduce(valReduction);
    if (res) {
        smem[threadIdx.y][0] = res;
        smem[threadIdx.y][2] = y * static_cast<T>(res);
        smem[threadIdx.y][5] = y2 * static_cast<T>(res);
        smem[threadIdx.y][9] = y3 * static_cast<T>(res);
        smem[threadIdx.y][1] = warpButterflyReduce(static_cast<T>(reductions[0]));
        smem[threadIdx.y][3] = warpButterflyReduce(static_cast<T>(reductions[1]));
        smem[threadIdx.y][6] = warpButterflyReduce(static_cast<T>(reductions[2]));
        smem[threadIdx.y][4] = smem[threadIdx.y][1] * y;
        smem[threadIdx.y][7] = smem[threadIdx.y][3] * y;
        smem[threadIdx.y][8] = smem[threadIdx.y][1] * y2;
    }
    __syncthreads();

    if (threadIdx.x < blockSizeY && threadIdx.y < nMoments)
        smem[threadIdx.y][0] = halfWarpButterflyReduce(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < nMoments) {
        if (smem[threadIdx.x][0])
            atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
    }
}

template <typename T>
__global__ void ComputeSpatialMomentsSharedFullReductionS5(const PtrStepSzb img, const bool binary, T* moments) {
    constexpr int nMoments = 10;
    const unsigned int y = blockIdx.x * blockDim.y + threadIdx.y;
    constexpr bool fourByteAligned = true;
    constexpr int offsetX = 0;
    const int alignedOffset = fourByteAligned ? 0 : 4 - offsetX;
    __shared__ T smem[32][nMoments];

    if (threadIdx.y < nMoments && threadIdx.x < 32)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();


    T reductions[4] = { 0 };
    if (y < img.rows) {


        //const uchar* rowPtr = &src[y * step];
        //// load uncoalesced head
        //if (!fourByteAligned && threadIdx.x == 0) {
        //    for (int x = 0; x < min(alignedOffset, cols); x++)
        //        Emulation::smem::atomicAdd(&shist[static_cast<int>(rowPtr[x])], 1);
        //}
        //const unsigned int img_index = y * img.step + x;
        //const unsigned int data = *((const unsigned int*)(&(img.data[img_index])));

        const unsigned int* rowPtrIntAligned = (const unsigned int*)(fourByteAligned ? img.ptr(y) : img.ptr(y) + alignedOffset);
        const int cols_4 = fourByteAligned ? img.cols / 4 : (img.cols - alignedOffset) / 4;
        for (int x = threadIdx.x; x < cols_4; x += blockDim.x) {
            //const unsigned int data = *((const unsigned int*)(&(rowPtrIntAligned+x)));
            const unsigned int data = rowPtrIntAligned[x];

            //const uchar el = ((data >> i * 8) & 0xFFU);
#pragma unroll 4
            for (int i = 0; i < 4; i++) {
                const int iX = 4 * x + i;
                const unsigned long x2 = iX * iX, x3 = x2 * iX;
                const uchar el = ((data >> i * 8) & 0xFFU);
                T val = (!binary || el == 0) ? el : 1;

                reductions[0] += val;
                //if (y == 14)
                //    printf("%f\n", val);
                reductions[1] += val * iX;
                reductions[2] += val * x2;
                reductions[3] += val * x3;

            }

            //const unsigned long x2 = x * x, x3 = x2 * x;
            //T val = (!binary || img(y, x) == 0) ? img(y, x) : 1;
            //reductions[0] += val;
            //reductions[1] += val * x;
            //reductions[2] += val * x2;
            //reductions[3] += val * x3;
        }



        //// load uncoalesced tail
        //if (threadIdx.x == 0) {
        //    const int iTailStart = fourByteAligned ? cols_4 * 4 : cols_4 * 4 + alignedOffset;
        //    for (int x = iTailStart; x < cols; x++)
        //        Emulation::smem::atomicAdd(&shist[static_cast<int>(rowPtr[x])], 1);
        //}

    }

    const unsigned long y2 = y * y, y3 = y2 * y;
    T res = warpButterflyReduce(static_cast<T>(reductions[0]));
    //if (y == 14)
    //    printf("%f\n", res);
    if (res) {
        smem[threadIdx.y][0] = res;
        smem[threadIdx.y][2] = y * res;
        smem[threadIdx.y][5] = y2 * res;
        smem[threadIdx.y][9] = y3 * res;

        smem[threadIdx.y][1] = warpButterflyReduce(reductions[1]);
        smem[threadIdx.y][3] = warpButterflyReduce(reductions[2]);
        smem[threadIdx.y][6] = warpButterflyReduce(reductions[3]);

        smem[threadIdx.y][4] = smem[threadIdx.y][1] * y;
        smem[threadIdx.y][7] = smem[threadIdx.y][3] * y;
        smem[threadIdx.y][8] = smem[threadIdx.y][1] * y2;
    }
    __syncthreads();

    if (threadIdx.x < 32 && threadIdx.y < nMoments)
        smem[threadIdx.y][0] = warpButterflyReduce(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < nMoments) {
        if (smem[threadIdx.x][0])
            atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
    }
}

template <typename T>
__global__ void ComputeSpatialMomentsSharedFullReductionCoaleced(const PtrStepSzb img, const bool binary, T* moments) {
    const unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    constexpr int nMoments = 10;
    __shared__ T smem[blockSizeY][nMoments];

    if (threadIdx.x < blockSizeY && threadIdx.y < nMoments)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();


    uchar val[4] = { 0 };
    if (y < img.rows && x < img.cols) {
        const unsigned int img_index = y * img.step + x;
        const unsigned int data = *((const unsigned int*)(&(img.data[img_index])));

        // could we read the unaligned head and tail here?
        // use threadIdx.x == 0 and loop?
        // would need to happen before we sum up the warp results - could work overly complicated, check benchmarks first

        // could perform the sum, would need to do every calc here first???

        // try to read into val array containing all then just work as normal?


        // needs to be here for all threads in a warp to be utilized.
#pragma unroll 4
        for (int i = 0; i < 4; i++) {
            const uchar el = ((data >> i * 8) & 0xFFU);
            val[i] = (!binary || el == 0) ? el : 1;
        }
    }

    const unsigned long y2 = y * y, y3 = y2 * y;
#pragma unroll 4
    for (int i = 0; i < 4; i++) {
        const int iX = x + i;
        const unsigned long x2 = iX * iX, x3 = x2 * iX;
        //printf("%f\n", static_cast<T>(val[i]));
        T res = warpButterflyReduce(static_cast<T>(val[i]));
        if (res) {
            //printf("%f\n", res);
            smem[threadIdx.y][0] += res;
            smem[threadIdx.y][1] += warpButterflyReduce(iX * static_cast<T>(val[i]));
            smem[threadIdx.y][3] += warpButterflyReduce(x2 * static_cast<T>(val[i]));
            smem[threadIdx.y][4] += warpButterflyReduce(iX * y * static_cast<T>(val[i]));
            smem[threadIdx.y][6] += warpButterflyReduce(x3 * static_cast<T>(val[i]));
            smem[threadIdx.y][7] += warpButterflyReduce(x2 * y * static_cast<T>(val[i]));
            smem[threadIdx.y][8] += warpButterflyReduce(iX * y2 * static_cast<T>(val[i]));
        }
    }

    if (smem[threadIdx.y][0]) {
        smem[threadIdx.y][2] = y * smem[threadIdx.y][0];
        smem[threadIdx.y][5] = y2 * smem[threadIdx.y][0];
        smem[threadIdx.y][9] = y3 * smem[threadIdx.y][0];
    }

    __syncthreads();

    if (threadIdx.x < blockSizeY && threadIdx.y < nMoments)
        smem[threadIdx.y][0] = halfWarpButterflyReduce(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < nMoments){
        if (smem[threadIdx.x][0])
            atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
        }
}





template <typename T>
__global__ void ComputeSpatialMomentsSharedFullReductionS1F(const cuda::PtrStepSzb img, bool binary, T* moments) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ T smem[16][10];

    if (threadIdx.y < 10)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    T val = 0;
    if (y < img.rows && x < img.cols) {
        const unsigned int img_index = y * img.step + x;
        val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;
    }

    const unsigned long x2 = x * x, x3 = x2 * x;
    const unsigned long y2 = y * y, y3 = y2 * y;
    T res = warpButterflyReduce(val);
    if (res) {
        if(threadIdx.x == 0)
            smem[threadIdx.y][0] = res;
        T tmp = warpButterflyReduce(x * val);
        if (threadIdx.x == 0) smem[threadIdx.y][1] = tmp;
        tmp = y * res;
        if (threadIdx.x == 0) smem[threadIdx.y][2] = tmp;
        tmp = warpButterflyReduce(x2 * val);
        if (threadIdx.x == 0) smem[threadIdx.y][3] = tmp;
        tmp = warpButterflyReduce(x * y * val);
        if (threadIdx.x == 0) smem[threadIdx.y][4] = tmp;
        tmp = y2 * res;
        if (threadIdx.x == 0) smem[threadIdx.y][5] = tmp;
        tmp = warpButterflyReduce(x3 * val);
        if (threadIdx.x == 0) smem[threadIdx.y][6] = tmp;
        tmp = warpButterflyReduce(x2 * y * val);
        if (threadIdx.x == 0) smem[threadIdx.y][7] = tmp;
        tmp = warpButterflyReduce(x * y2 * val);
        if (threadIdx.x == 0) smem[threadIdx.y][8] = tmp;
        tmp = y3 * res;
        if (threadIdx.x == 0) smem[threadIdx.y][9] = tmp;
    }
    __syncthreads();

    // blockSizeY - this has to be 16 as below we are using half warp reduce and above we have to enforce 32 for warp reduce
    if (threadIdx.x < blockSizeY && threadIdx.y < 10) {
        T tmp = halfWarpButterflyReduce(smem[threadIdx.x][threadIdx.y]);
        if(threadIdx.x == 0) smem[threadIdx.y][0] = tmp;
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < 10)
        atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
}


template <typename T>
__global__ void ComputeCentralMomentsSharedUchar(const cuda::PtrStepSzb img, bool binary, const T* m00, const T* m10, const T* m01, T* moments) {

    if (*m00 == 0 || *m10 == 0 || *m01 == 0)
        return;
    const T cX = *m10 / *m00;
    const T cY = *m01 / *m00;
    //if (!cX && !cY) return;

    const unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x)*4;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    constexpr int n = 7;
    __shared__ T smem[16][n];

    if (threadIdx.y < n && threadIdx.x < 16)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    uchar val[4] = { 0 };
    if (y < img.rows && x < img.cols) {
        //const unsigned int img_index = y * img.step + x;
        //val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;

        const unsigned int img_index = y * img.step + x;
        const unsigned int data = *((const unsigned int*)(&(img.data[img_index])));
#pragma unroll 4
        for (int i = 0; i < 4; i++) {
            const uchar el = ((data >> i * 8) & 0xFFU);
            val[i] = (!binary || el == 0) ? el : 1;
        }
    }


    const T y1 = y - cY, y2 = y1 * y1, y3 = y2 * y1;
    T resSum = 0;
#pragma unroll 4
    for (int i = 0; i < 4; i++) {
        const int iX = x + i;
        const T x1 = iX - cX, x2 = x1 * x1, x3 = x2 * x1;
        T res = warpButterflyReduce(static_cast<T>(val[i]));
        if (res) {
            resSum += res;
            smem[threadIdx.y][0] += warpButterflyReduce(x2 * static_cast<T>(val[i]));
            smem[threadIdx.y][1] += warpButterflyReduce(x1 * y1 * static_cast<T>(val[i]));
            smem[threadIdx.y][3] += warpButterflyReduce(x3 * static_cast<T>(val[i]));
            smem[threadIdx.y][4] += warpButterflyReduce(x2 * y1 * static_cast<T>(val[i]));
            smem[threadIdx.y][5] += warpButterflyReduce(x1 * y2 * static_cast<T>(val[i]));
        }
    }

    if (resSum) {
        smem[threadIdx.y][2] = y2 * resSum;
        smem[threadIdx.y][6] = y3 * resSum;
    }
    __syncthreads();

    //smem[threadIdx.y][0] = warpButterflyReduce(x2 * val);
    //smem[threadIdx.y][1] = warpButterflyReduce(x1 * y1 * val);
    //smem[threadIdx.y][2] = y2 * res;
    //smem[threadIdx.y][3] = warpButterflyReduce(x3 * val);
    //smem[threadIdx.y][4] = warpButterflyReduce(x2 * y1 * val);
    //smem[threadIdx.y][5] = warpButterflyReduce(x1 * y2 * val);
    //smem[threadIdx.y][6] = y3 * res;

    // blockSizeY - this has to be 16 as below we are using half warp reduce and above we have to enforce 32 for warp reduce
    if (threadIdx.x < blockSizeY && threadIdx.y < n)
        smem[threadIdx.y][0] = halfWarpButterflyReduce(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < n)
        atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
}

template <typename T>
__global__ void ComputeSpatialMomentsSharedPartialReduction(const cuda::PtrStepSzb img, bool binary, T* moments) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ T smem[10];

    if (threadIdx.y == 0 && threadIdx.x < 10)
        smem[threadIdx.x] = 0;
    __syncthreads();

    T val = 0;
    if (y < img.rows && x < img.cols) {
        const unsigned int img_index = y * img.step + x;
        val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;
    }

    const T m00 = warpButterflyReduce(val);
    T m10 = 0, m01 = 0, m20 = 0, m11 = 0, m02 = 0, m30 = 0, m21 = 0, m12 = 0, m03 = 0;
    const unsigned long x2 = x * x, x3 = x2 * x;
    const unsigned long y2 = y * y, y3 = y2 * y;
    if (m00) {
        m10 = warpButterflyReduce(x * val);
        m01 = y * m00;
        m20 = warpButterflyReduce(x2 * val);
        m11 = warpButterflyReduce(x * y * val);
        m02 = y2 * m00;
        m30 = warpButterflyReduce(x3 * val);
        m21 = warpButterflyReduce(x2 * y * val);
        m12 = warpButterflyReduce(x * y2 * val);
        m03 = y3 * m00;
    }

    if (threadIdx.x == 0) {
        atomicAdd(&smem[0], m00);
        atomicAdd(&smem[1], m10);
        atomicAdd(&smem[2], m01);
        atomicAdd(&smem[3], m20);
        atomicAdd(&smem[4], m11);
        atomicAdd(&smem[5], m02);
        atomicAdd(&smem[6], m30);
        atomicAdd(&smem[7], m21);
        atomicAdd(&smem[8], m12);
        atomicAdd(&smem[9], m03);
    }
    __syncthreads();


    if (threadIdx.y == 0 && threadIdx.x < 10)
        atomicAdd(&moments[threadIdx.x], smem[threadIdx.x]);
}

template <typename T>
__global__ void ComputeCentralMomentsShared1(const cuda::PtrStepSzb img, bool binary, T* centroid, T* moments) {


    //if (*m00 == 0 || *m10 == 0 || *m01 == 0)
    //    return;

    const T cX = centroid[0];
    const T cY = centroid[1];

    //if (!cX && !cY) return;

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    //const T x1 = x - cX, y1 = y - cY;
    constexpr int n = 7;
    __shared__ T smem[16][n];

    if (threadIdx.y < n && threadIdx.x < 16)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    uchar val = 0;
    if (y < img.rows && x < img.cols) {
        const unsigned int img_index = y * img.step + x;
        val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;
    }

    //const unsigned long x2 = x * x, x3 = x2 * x;
    //const unsigned long y2 = y * y, y3 = y2 * y;
    T res = warpButterflyReduce(static_cast<T>(val));
    if (res) {
        const T x1 = x - cX, x2 = x1 * x1, x3 = x2 * x1; // might be quicker to do x1*x1*x1, let the compiler decide?
        const T y1 = y - cY, y2 = y1 * y1, y3 = y2 * y1;
        smem[threadIdx.y][0] = warpButterflyReduce(x2 * static_cast<T>(val));
        smem[threadIdx.y][1] = warpButterflyReduce(x1 * y1 * static_cast<T>(val));
        smem[threadIdx.y][2] = y2 * res;
        smem[threadIdx.y][3] = warpButterflyReduce(x3 * static_cast<T>(val));
        smem[threadIdx.y][4] = warpButterflyReduce(x2 * y1 * static_cast<T>(val));
        smem[threadIdx.y][5] = warpButterflyReduce(x1 * y2 * static_cast<T>(val));
        smem[threadIdx.y][6] = y3 * res;
    }
    __syncthreads();

    // blockSizeY - this has to be 16 as below we are using half warp reduce and above we have to enforce 32 for warp reduce
    if (threadIdx.x < blockSizeY && threadIdx.y < n)
        smem[threadIdx.y][0] = halfWarpButterflyReduce(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < n) {
        const T tmp = smem[threadIdx.x][0];
        if (tmp)
            atomicAdd(&moments[threadIdx.x], tmp);
    }
    //atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
}




template <typename T>
__global__ void ComputeCentralMomentsShared(const cuda::PtrStepSzb img, bool binary, const T* m00, const T* m10, const T* m01, T* moments) {


    //if (*m00 == 0 || *m10 == 0 || *m01 == 0)
    //    return;

    const T cX = *m10 / *m00;
    const T cY = *m01 / *m00;
    if (!cX && !cY) return;

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    //const T x1 = x - cX, y1 = y - cY;
    constexpr int n = 7;
    __shared__ T smem[16][n];

    if (threadIdx.y < n && threadIdx.x < 16)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    uchar val = 0;
    if (y < img.rows && x < img.cols) {
        const unsigned int img_index = y * img.step + x;
        val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;
    }

    //const unsigned long x2 = x * x, x3 = x2 * x;
    //const unsigned long y2 = y * y, y3 = y2 * y;
    T res = warpButterflyReduce(static_cast<T>(val));
    if (res) {
        const T x1 = x - cX, x2 = x1 * x1, x3 = x2 * x1;
        const T y1 = y - cY, y2 = y1 * y1, y3 = y2 * y1;
        smem[threadIdx.y][0] = warpButterflyReduce(x2 * static_cast<T>(val));
        smem[threadIdx.y][1] = warpButterflyReduce(x1 * y1 * static_cast<T>(val));
        smem[threadIdx.y][2] = y2 * res;
        smem[threadIdx.y][3] = warpButterflyReduce(x3 * static_cast<T>(val));
        smem[threadIdx.y][4] = warpButterflyReduce(x2 * y1 * static_cast<T>(val));
        smem[threadIdx.y][5] = warpButterflyReduce(x1 * y2 * static_cast<T>(val));
        smem[threadIdx.y][6] = y3 * res;
    }
    __syncthreads();

    // blockSizeY - this has to be 16 as below we are using half warp reduce and above we have to enforce 32 for warp reduce
    if (threadIdx.x < blockSizeY && threadIdx.y < n)
        smem[threadIdx.y][0] = halfWarpButterflyReduce(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < n) {
        const T tmp = smem[threadIdx.x][0];
        if(tmp)
            atomicAdd(&moments[threadIdx.x], tmp);
    }
        //atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][0]);
}


template<typename T>
__global__ void ComputeCenteroid1(T* moments) {
    moments[17] = moments[1] / moments[0];
    moments[18] = moments[2] / moments[0];
}



__global__ void ComputeCenteroid(const double* moments, double2* centroid) {
    centroid->x = moments[1] / moments[0];
    centroid->y = moments[2] / moments[0];
}

__global__ void ComputeCenteralMoments(const cuda::PtrStepSzb img, bool binary,
                                       const double2* centroid, double* moments) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < img.rows && x < img.cols) {
      const unsigned int img_index = y * img.step + x;
      const unsigned char val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;
      if (val > 0) {
        const double x1 = x - centroid->x, x2 = x1 * x1, x3 = x2 * x1;
        const double y1 = y - centroid->y, y2 = y1 * y1, y3 = y2 * y1;

        atomicAdd(&moments[10], x2      * val);
        atomicAdd(&moments[11], x1 * y1 * val);
        atomicAdd(&moments[12],      y2 * val);
        atomicAdd(&moments[13], x3      * val);
        atomicAdd(&moments[14], x2 * y1 * val);
        atomicAdd(&moments[15], x1 * y2 * val);
        atomicAdd(&moments[16],      y3 * val);
      }
    }
}

//void ComputeCenteralNormalizedMoments(cv::Moments& moments_cpu) {
//    const double m00_pow2 = pow(moments_cpu.m00, 2), m00_pow2p5 = pow(moments_cpu.m00, 2.5);
//
//    moments_cpu.nu20 = moments_cpu.mu20 / m00_pow2;
//    moments_cpu.nu11 = moments_cpu.mu11 / m00_pow2;
//    moments_cpu.nu02 = moments_cpu.mu02 / m00_pow2;
//    moments_cpu.nu30 = moments_cpu.mu30 / m00_pow2p5;
//    moments_cpu.nu21 = moments_cpu.mu21 / m00_pow2p5;
//    moments_cpu.nu12 = moments_cpu.mu12 / m00_pow2p5;
//    moments_cpu.nu03 = moments_cpu.mu03 / m00_pow2p5;
//}

//void Benchmark(const int idx, const cv::cuda::GpuMat& img, bool binary) {
//    dim3 blockSize(blockSizeX, blockSizeY, 1);
//    dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y, 1);
//    cuda::Stream stream;
//    cuda::Event start, end;
//
//    // calculate gs result
//    GpuMat momentsGpuGs = GpuMat(1, momentsSize, CV_64F, cv::Scalar(0));
//    double2* centroid;
//    cudaSafeCall(cudaMalloc(&centroid, sizeof(double2)));
//    start.record(stream);
//    ComputeSpatialMoments << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpuGs.ptr<double>(0));
//    ComputeCenteroid << <dim3(1, 1, 1), dim3(1, 1, 1), 0, cuda::StreamAccessor::getStream(stream) >> > (momentsGpuGs.ptr<double>(0), centroid);
//    ComputeCenteralMoments << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, centroid, momentsGpuGs.ptr<double>(0));
//    end.record(stream);
//    stream.waitForCompletion();
//    const float nsGs = Event::elapsedTime(start, end) * 1000;
//    Mat momentsCpuGs; momentsGpuGs.download(momentsCpuGs);
//    cudaSafeCall(cudaFree(centroid));
//
//    GpuMat momentsGpu, momentsGpu64F;
//    switch (idx) {
//        case 0:
//        {
//            printf("\nOriginal using double\n");
//            momentsGpu = GpuMat(1, momentsSize, CV_64F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMoments << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0));
//            end.record(stream);
//            momentsGpu64F = momentsGpu;
//            break;
//        }
//        case 1:
//        {
//            printf("\nOriginal using float\n");
//            momentsGpu = GpuMat(1, momentsSize, CV_32F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMoments << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0));
//            end.record(stream);
//            momentsGpu.convertTo(momentsGpu64F, CV_64F);
//            break;
//        }
//        case 2:
//        {
//            printf("\nShared memory with partial reduction using double\n");
//            momentsGpu = GpuMat(1, momentsSize, CV_64F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMomentsSharedPartialReduction << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0));
//            end.record(stream);
//            momentsGpu64F = momentsGpu;
//            break;
//        }
//        case 3:
//        {
//            printf("\nShared memory with partial reduction using float\n");
//            momentsGpu = GpuMat(1, momentsSize, CV_32F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMomentsSharedPartialReduction << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0));
//            end.record(stream);
//            momentsGpu.convertTo(momentsGpu64F, CV_64F);
//            break;
//        }
//        case 4:
//        {
//            printf("\nShared memory with full reduction using double\n");
//            momentsGpu = GpuMat(1, momentsSize, CV_64F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMomentsSharedFullReduction << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0));
//            end.record(stream);
//            momentsGpu64F = momentsGpu;
//            break;
//        }
//        case 5:
//        {
//            printf("\nShared memory with full reduction using float\n");
//            momentsGpu = GpuMat(1, momentsSize, CV_32F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMomentsSharedFullReduction << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0));
//            end.record(stream);
//            momentsGpu.convertTo(momentsGpu64F, CV_64F);
//            break;
//        }
//        case 6:
//        {
//            printf("\nShared memory with full reduction using float S1\n");
//            momentsGpu = GpuMat(1, momentsSize, CV_32F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMomentsSharedFullReductionS1 << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0));
//            end.record(stream);
//            momentsGpu.convertTo(momentsGpu64F, CV_64F);
//            break;
//        }
//        case 7:
//        {
//            printf("\nCentral Moments Shared memory with full reduction using double\n");
//            momentsGpu = GpuMat(1, momentsSize, CV_64F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMomentsSharedFullReductionS1 << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0));
//
//            // should we pass pointer or ptrstepsz?
//            ComputeCenteroid1 << < dim3(1, 1, 1), dim3(1, 1, 1), 0, cuda::StreamAccessor::getStream(stream) >> > (momentsGpu.ptr<double>(0));
//            //ComputeCentralMomentsShared << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0), momentsGpu.ptr<double>(0)+1, momentsGpu.ptr<double>(0)+2, momentsGpu.ptr<double>(0)+10);
//            ComputeCentralMomentsShared1 << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0) + 17, momentsGpu.ptr<double>(0) + 10);
//            end.record(stream);
//            //momentsGpu.convertTo(momentsGpu64F, CV_64F);
//            momentsGpu64F = momentsGpu;
//            //momentsGpu.convertTo(momentsGpu64F, CV_64F);
//            break;
//        }
//        case 8:
//        {
//            printf("\nCentral Moments Shared memory with full reduction using float\n");
//            momentsGpu = GpuMat(1, momentsSize, CV_32F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMomentsSharedFullReductionS1 << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0));
//            ComputeCentralMomentsShared << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0), momentsGpu.ptr<float>(0) + 1, momentsGpu.ptr<float>(0) + 2, momentsGpu.ptr<float>(0) + 10);
//            end.record(stream);
//            momentsGpu.convertTo(momentsGpu64F, CV_64F);
//            break;
//        }
//        case 9:
//        {
//            printf("\nShared memory with full reduction using double and coalecsed reads\n");
//            blockSize = dim3(blockSizeX, blockSizeY, 1);
//            gridSize = dim3(divUp(img.cols/4, blockSizeX), divUp(img.rows, blockSizeY));
//            momentsGpu = GpuMat(1, momentsSize, CV_64F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMomentsSharedFullReductionCoaleced << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0));
//            end.record(stream);
//            momentsGpu64F = momentsGpu;
//            break;
//        }
//        case 10:
//        {
//            printf("\nShared memory with full reduction using float and coalecsed reads\n");
//            blockSize = dim3(blockSizeX, blockSizeY, 1);
//            gridSize = dim3(divUp(img.cols / 4, blockSizeX), divUp(img.rows, blockSizeY));
//            momentsGpu = GpuMat(1, momentsSize, CV_32F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMomentsSharedFullReductionCoaleced << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0));
//            end.record(stream);
//            momentsGpu.convertTo(momentsGpu64F, CV_64F);
//            break;
//        }
//        case 11:
//        {
//            printf("\nCentral Moments memory with full coalesced reduction using double\n");
//            blockSize = dim3(blockSizeX, blockSizeY, 1);
//            gridSize = dim3(divUp(img.cols / 4, blockSizeX), divUp(img.rows, blockSizeY));
//            momentsGpu = GpuMat(1, momentsSize, CV_64F, cv::Scalar(0));
//            start.record(stream);
//            ComputeSpatialMomentsSharedFullReductionCoaleced << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0));
//            ComputeCentralMomentsSharedUchar << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0), momentsGpu.ptr<double>(0) + 1, momentsGpu.ptr<double>(0) + 2, momentsGpu.ptr<double>(0) + 10);
//            end.record(stream);
//            momentsGpu64F = momentsGpu;
//            break;
//        }
//        case 12:
//        {
//            printf("\nCentral Moments Shared memory with full coalesced reduction using float\n");
//            blockSize = dim3(blockSizeX, blockSizeY, 1);
//            gridSize = dim3(divUp(img.cols / 4, blockSizeX), divUp(img.rows, blockSizeY));
//            momentsGpu = GpuMat(1, momentsSize, CV_32F, cv::Scalar(0));
//            start.record(stream);
//            //ComputeSpatialMomentsSharedFullReductionS1 << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0));
//
//            ComputeSpatialMomentsSharedFullReductionCoaleced << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0));
//            ComputeCentralMomentsSharedUchar << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0), momentsGpu.ptr<float>(0) + 1, momentsGpu.ptr<float>(0) + 2, momentsGpu.ptr<float>(0) + 10);
//            end.record(stream);
//            momentsGpu.convertTo(momentsGpu64F, CV_64F);
//            break;
//        }
//    }
//
//    stream.waitForCompletion();
//    const float ns = Event::elapsedTime(start, end)*1000;
//    printf("  eltime - %.2fus (GS: %.2fus), speedup %.2fX\n", ns, nsGs, nsGs/ns);
//
//    Mat momentsCpu; momentsGpu64F.download(momentsCpu);
//    double cumErr = 0;
//    for (int i = 0; i < 17; i++) {
//        printf("%f, %f\n", momentsCpuGs.at<double>(i), momentsCpu.at<double>(i));
//        cumErr += abs(momentsCpuGs.at<double>(i) - momentsCpu.at<double>(i));
//    }
//    if (cumErr != 0)
//        printf("  cumulative error %f\n", cumErr);
//
//}

//enum MomentType {
//    SPATIAL,
//    CENTRAL
//};

template <typename T>
void moments(const PtrStepSzb src, PtrStep<T> moments, const bool binary, const bool mixedPrecision, const int offsetX, const cudaStream_t stream) {
    dim3 blockSize(blockSizeX, blockSizeY);
    //dim3 gridSize(divUp(src.cols, blockSizeX), divUp(src.rows, blockSizeY));
    // heuristic
    // for double don't use
    // for float use - minimal advantage -> check smaller
    // add mixed precision - experimental option
    // + above heuristic -> and host return, then run bench.



    //ComputeSpatialMoments << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());
    //ComputeSpatialMomentsSharedFullReductionS1 << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());
    //ComputeSpatialMomentsSharedFullReductionS2 << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());
    dim3 gridSize = dim3(divUp(src.rows, blockSizeY));
    //ComputeSpatialMomentsSharedFullReductionS3 << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());
    //if(binary)


        //ComputeSpatialMomentsSharedFullReductionS3a<T> << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());
    //else
    //    ComputeSpatialMomentsSharedFullReductionS3a<false> << <gridSize, blockSize, 0, stream >> > (src, moments.ptr());

    // for double - if double can use extra mixed precision.


    // for float
    // S4 (coalesced is slower on double with offsets)
    if(offsetX)
        spatialMoments<float, true, false> << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr(), offsetX);
        //ComputeSpatialMomentsSharedFullReductionS4<false, T> << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr(), offsetX);
    else
        spatialMoments<float, true, true> << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());
        //ComputeSpatialMomentsSharedFullReductionS4<true, T> << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());

    //blockSize = dim3(blockSizeX, 32);
    //gridSize = dim3(divUp(src.rows, blockSizeY));
    //ComputeSpatialMomentsSharedFullReductionS5 << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());




    //gridSize = dim3(divUp(src.cols/4, blockSizeX), divUp(src.rows, blockSizeY));
    //ComputeSpatialMomentsSharedFullReductionCoaleced << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());
    //if (computeCentral) {
    //    //ComputeCentralMomentsShared << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr(), moments.ptr() + 1, moments.ptr() + 2, moments.ptr() + 10);
    //    ComputeCenteroid1 << < dim3(1, 1, 1), dim3(1, 1, 1), 0, stream >> > (moments.ptr());
    //    //ComputeCentralMomentsShared << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0), momentsGpu.ptr<double>(0)+1, momentsGpu.ptr<double>(0)+2, momentsGpu.ptr<double>(0)+10);
    //    ComputeCentralMomentsShared1 << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr()+17, moments.ptr()+10);
    //}

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());

    // moments can be float or double
    // can we request calculation to be float for spatial and double for the other
    // need to request type of calc

    // if spatial just do one
    // central both

    // all - need to normalize forget
    // need a helper routine which downloads and normalizes the result


}

template <>
void moments(const PtrStepSzb src, PtrStep<double> moments, const bool binary, const bool mixedPrecision, const int offsetX, const cudaStream_t stream) {
    dim3 blockSize(blockSizeX, blockSizeY);
    dim3 gridSize = dim3(divUp(src.rows, blockSizeY));

    if(mixedPrecision)
        spatialMoments<double, false, false, float> << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());
    else
        spatialMoments<double> << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());
    //ComputeSpatialMomentsSharedFullReductionS3a<double> << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}

template void moments<float>(const PtrStepSzb src, PtrStep<float> moments, const bool binary, const bool mixedPrecision, const int offsetX, const cudaStream_t stream);
//template void moments<double>(const PtrStepSzb src, PtrStep<double> moments, const bool binary, const int offsetX, const cudaStream_t stream);

//void Moments(const cv::cuda::GpuMat& img, bool binary) {
//
//    const dim3 blockSize(blockSizeX, blockSizeY, 1);
//    const dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x,
//                        (img.rows + blockSize.y - 1) / blockSize.y, 1);
//
//    double2* centroid;
//    cudaSafeCall(cudaMalloc(&centroid, sizeof(double2)));
//    cv::cuda::GpuMat moments_gpu(1, momentsSize, CV_64F, cv::Scalar(0));
//    ComputeSpatialMoments <<<gridSize, blockSize>>>(img, binary, moments_gpu.ptr<double>(0));
//    cudaSafeCall(cudaGetLastError());
//
//    ComputeCenteroid <<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(moments_gpu.ptr<double>(0), centroid);
//    cudaSafeCall(cudaGetLastError());
//
//    ComputeCenteralMoments <<<gridSize, blockSize>>>(img, binary, centroid, moments_gpu.ptr<double>(0));
//    cudaSafeCall(cudaFree(centroid));
//    cudaSafeCall(cudaGetLastError());
//
//    //cv::Moments moments_cpu;
//    //cv::Mat moments_map(1, momentsSize, CV_64F, reinterpret_cast<double*>(&moments_cpu));
//    //moments_gpu.download(moments_map);
//    //cudaSafeCall(cudaDeviceSynchronize());
//
//    //ComputeCenteralNormalizedMoments(moments_cpu);
//
//    //for (int i = 0; i < 11; i++)
//    //    Benchmark(i, img, binary);
//
//    //Benchmark(8, img, binary);
//    ////Benchmark(12, img, binary);
//    //Benchmark(7, img, binary);
//    //Benchmark(4, img, binary);
//    //return moments_cpu;
//}

}}}}


#endif /* CUDA_DISABLER */
