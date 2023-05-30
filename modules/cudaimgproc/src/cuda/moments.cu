// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/emulation.hpp"
#include "opencv2/core/cuda/transform.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/utility.hpp"
#include "opencv2/core/cuda.hpp"
#include <opencv2/core/cuda_stream_accessor.hpp>

using namespace cv::cuda;
using namespace cv::cuda::device;


namespace cv { namespace cuda { namespace device { namespace imgproc {

constexpr int blockSizeX = 32;
constexpr int blockSizeY = 16;
constexpr int momentsSize = sizeof(cv::Moments) / sizeof(double);

constexpr int m00 = offsetof(cv::Moments, m00) / sizeof(double);
constexpr int m10 = offsetof(cv::Moments, m10) / sizeof(double);
constexpr int m01 = offsetof(cv::Moments, m01) / sizeof(double);
constexpr int m20 = offsetof(cv::Moments, m20) / sizeof(double);
constexpr int m11 = offsetof(cv::Moments, m11) / sizeof(double);
constexpr int m02 = offsetof(cv::Moments, m02) / sizeof(double);
constexpr int m30 = offsetof(cv::Moments, m30) / sizeof(double);
constexpr int m21 = offsetof(cv::Moments, m21) / sizeof(double);
constexpr int m12 = offsetof(cv::Moments, m12) / sizeof(double);
constexpr int m03 = offsetof(cv::Moments, m03) / sizeof(double);

constexpr int mu20 = offsetof(cv::Moments, mu20) / sizeof(double);
constexpr int mu11 = offsetof(cv::Moments, mu11) / sizeof(double);
constexpr int mu02 = offsetof(cv::Moments, mu02) / sizeof(double);
constexpr int mu30 = offsetof(cv::Moments, mu30) / sizeof(double);
constexpr int mu21 = offsetof(cv::Moments, mu21) / sizeof(double);
constexpr int mu12 = offsetof(cv::Moments, mu12) / sizeof(double);
constexpr int mu03 = offsetof(cv::Moments, mu03) / sizeof(double);

__global__ void ComputeSpatialMoments(const cuda::PtrStepSzb img, bool binary, double* moments) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < img.rows && x < img.cols) {
      const unsigned int img_index = y * img.step + x;
      const unsigned char val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;
      if (val > 0) {
        const unsigned long x2 = x * x, x3 = x2 * x;
        const unsigned long y2 = y * y, y3 = y2 * y;

        atomicAdd(&moments[m00],           val);
        atomicAdd(&moments[m10], x       * val);
        atomicAdd(&moments[m01],      y  * val);
        atomicAdd(&moments[m20], x2      * val);
        atomicAdd(&moments[m11], x  * y  * val);
        atomicAdd(&moments[m02],      y2 * val);
        atomicAdd(&moments[m30], x3      * val);
        atomicAdd(&moments[m21], x2 * y  * val);
        atomicAdd(&moments[m12], x  * y2 * val);
        atomicAdd(&moments[m03],      y3 * val);
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

            atomicAdd(&moments[m00], val);
            atomicAdd(&moments[m10], x * val);
            atomicAdd(&moments[m01], y * val);
            atomicAdd(&moments[m20], x2 * val);
            atomicAdd(&moments[m11], x * y * val);
            atomicAdd(&moments[m02], y2 * val);
            atomicAdd(&moments[m30], x3 * val);
            atomicAdd(&moments[m21], x2 * y * val);
            atomicAdd(&moments[m12], x * y2 * val);
            atomicAdd(&moments[m03], y3 * val);
        }
    }
}

template <typename T>
__device__ T warpButterflyReduce(T value) {
    for (int i = 16; i >= 1; i /= 2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
    return value;
}

template <typename T>
__device__ T halfWarpButterflyReduce(T value) {
    for (int i = 8; i >= 1; i /= 2)
        value += __shfl_xor_sync(0xffff, value, i, 32);
    return value;
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
        smem[2][threadIdx.y] = warpButterflyReduce(y * val);
        smem[3][threadIdx.y] = warpButterflyReduce(x2 * val);
        smem[4][threadIdx.y] = warpButterflyReduce(x * y * val);
        smem[5][threadIdx.y] = warpButterflyReduce(y2 * val);
        smem[6][threadIdx.y] = warpButterflyReduce(x3 * val);
        smem[7][threadIdx.y] = warpButterflyReduce(x2 * y * val);
        smem[8][threadIdx.y] = warpButterflyReduce(x * y2 * val);
        smem[9][threadIdx.y] = warpButterflyReduce(y3 * val);
    }
    __syncthreads();

    if (threadIdx.x < blockSizeY && threadIdx.y < 10)
        smem[threadIdx.y][0] = halfWarpButterflyReduce(smem[threadIdx.y][threadIdx.x]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < 10)
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
        m01 = warpButterflyReduce(y * val);
        m20 = warpButterflyReduce(x2 * val);
        m11 = warpButterflyReduce(x * y * val);
        m02 = warpButterflyReduce(y2 * val);
        m30 = warpButterflyReduce(x3 * val);
        m21 = warpButterflyReduce(x2 * y * val);
        m12 = warpButterflyReduce(x * y2 * val);
        m03 = warpButterflyReduce(y3 * val);
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

__global__ void ComputeCenteroid(const double* moments, double2* centroid) {
    centroid->x = moments[m10] / moments[m00];
    centroid->y = moments[m01] / moments[m00];
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

        atomicAdd(&moments[mu20], x2      * val);
        atomicAdd(&moments[mu11], x1 * y1 * val);
        atomicAdd(&moments[mu02],      y2 * val);
        atomicAdd(&moments[mu30], x3      * val);
        atomicAdd(&moments[mu21], x2 * y1 * val);
        atomicAdd(&moments[mu12], x1 * y2 * val);
        atomicAdd(&moments[mu03],      y3 * val);
      }
    }
}

void ComputeCenteralNormalizedMoments(cv::Moments& moments_cpu) {
    const double m00_pow2 = pow(moments_cpu.m00, 2), m00_pow2p5 = pow(moments_cpu.m00, 2.5);

    moments_cpu.nu20 = moments_cpu.mu20 / m00_pow2;
    moments_cpu.nu11 = moments_cpu.mu11 / m00_pow2;
    moments_cpu.nu02 = moments_cpu.mu02 / m00_pow2;
    moments_cpu.nu30 = moments_cpu.mu30 / m00_pow2p5;
    moments_cpu.nu21 = moments_cpu.mu21 / m00_pow2p5;
    moments_cpu.nu12 = moments_cpu.mu12 / m00_pow2p5;
    moments_cpu.nu03 = moments_cpu.mu03 / m00_pow2p5;
}

void Benchmark(const int idx, const cv::cuda::GpuMat& img, bool binary) {
    const dim3 blockSize(blockSizeX, blockSizeY, 1);
    const dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y, 1);
    cuda::Stream stream;
    cuda::Event start, end;

    // calculate gs result
    GpuMat momentsGpuGs = GpuMat(1, momentsSize, CV_64F, cv::Scalar(0));
    start.record(stream);
    ComputeSpatialMoments << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpuGs.ptr<double>(0));
    end.record(stream);
    stream.waitForCompletion();
    const float nsGs = Event::elapsedTime(start, end) * 1000;
    Mat momentsCpuGs; momentsGpuGs.download(momentsCpuGs);

    GpuMat momentsGpu, momentsGpu64F;
    switch (idx) {
        case 0:
        {
            printf("\nOriginal using double\n");
            momentsGpu = GpuMat(1, momentsSize, CV_64F, cv::Scalar(0));
            start.record(stream);
            ComputeSpatialMoments << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0));
            end.record(stream);
            momentsGpu64F = momentsGpu;
            break;
        }
        case 1:
        {
            printf("\nOriginal using float\n");
            momentsGpu = GpuMat(1, momentsSize, CV_32F, cv::Scalar(0));
            start.record(stream);
            ComputeSpatialMoments << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0));
            end.record(stream);
            momentsGpu.convertTo(momentsGpu64F, CV_64F);
            break;
        }
        case 2:
        {
            printf("\nShared memory with partial reduction using double\n");
            momentsGpu = GpuMat(1, momentsSize, CV_64F, cv::Scalar(0));
            start.record(stream);
            ComputeSpatialMomentsSharedPartialReduction << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0));
            end.record(stream);
            momentsGpu64F = momentsGpu;
            break;
        }
        case 3:
        {
            printf("\nShared memory with partial reduction using float\n");
            momentsGpu = GpuMat(1, momentsSize, CV_32F, cv::Scalar(0));
            start.record(stream);
            ComputeSpatialMomentsSharedPartialReduction << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0));
            end.record(stream);
            momentsGpu.convertTo(momentsGpu64F, CV_64F);
            break;
        }
        case 4:
        {
            printf("\nShared memory with full reduction using double\n");
            momentsGpu = GpuMat(1, momentsSize, CV_64F, cv::Scalar(0));
            start.record(stream);
            ComputeSpatialMomentsSharedFullReduction << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<double>(0));
            end.record(stream);
            momentsGpu64F = momentsGpu;
            break;
        }
        case 5:
        {
            printf("\nShared memory with full reduction using float\n");
            momentsGpu = GpuMat(1, momentsSize, CV_32F, cv::Scalar(0));
            start.record(stream);
            ComputeSpatialMomentsSharedFullReduction << <gridSize, blockSize, 0, cuda::StreamAccessor::getStream(stream) >> > (img, binary, momentsGpu.ptr<float>(0));
            end.record(stream);
            momentsGpu.convertTo(momentsGpu64F, CV_64F);
            break;
        }
    }

    stream.waitForCompletion();
    const float ns = Event::elapsedTime(start, end)*1000;
    printf("  eltime - %.2fus (GS: %.2fus), speedup %.2fX\n", ns, nsGs, nsGs/ns);

    Mat momentsCpu; momentsGpu64F.download(momentsCpu);
    double cumErr = 0;
    for (int i = 0; i < 10; i++) {
        //printf("%f, %f\n", momentsCpuGs.at<double>(i), momentsCpu.at<double>(i));
        cumErr += abs(momentsCpuGs.at<double>(i) - momentsCpu.at<double>(i));
    }
    if (cumErr != 0)
        printf("  cumulative error %f\n", cumErr);

}

cv::Moments Moments(const cv::cuda::GpuMat& img, bool binary) {
    const dim3 blockSize(blockSizeX, blockSizeY, 1);
    const dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x,
                        (img.rows + blockSize.y - 1) / blockSize.y, 1);

    double2* centroid;
    cudaSafeCall(cudaMalloc(&centroid, sizeof(double2)));
    cv::cuda::GpuMat moments_gpu(1, momentsSize, CV_64F, cv::Scalar(0));
    ComputeSpatialMoments <<<gridSize, blockSize>>>(img, binary, moments_gpu.ptr<double>(0));
    cudaSafeCall(cudaGetLastError());

    ComputeCenteroid <<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(moments_gpu.ptr<double>(0), centroid);
    cudaSafeCall(cudaGetLastError());

    ComputeCenteralMoments <<<gridSize, blockSize>>>(img, binary, centroid, moments_gpu.ptr<double>(0));
    cudaSafeCall(cudaFree(centroid));
    cudaSafeCall(cudaGetLastError());

    cv::Moments moments_cpu;
    cv::Mat moments_map(1, momentsSize, CV_64F, reinterpret_cast<double*>(&moments_cpu));
    moments_gpu.download(moments_map);
    cudaSafeCall(cudaDeviceSynchronize());

    ComputeCenteralNormalizedMoments(moments_cpu);

    for (int i = 0; i < 6; i++)
        Benchmark(i, img, binary);

    return moments_cpu;
}

}}}}


#endif /* CUDA_DISABLER */
