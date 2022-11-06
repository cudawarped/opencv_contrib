// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CUDEV_PTR2D_TEXTURE_OBJECT_HPP
#define OPENCV_CUDEV_PTR2D_TEXTURE_OBJECT_HPP

#include <cstring>
#include "../common.hpp"
#include "glob.hpp"
#include "gpumat.hpp"
#include "traits.hpp"
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/utils/logger.hpp>

//  NEED COMMENT IN EXISTING TEXTURE THAT UNSAFE TO PASS TO KERNELS SEE EXAMPLE USAGE ....

/** \file texture_object.hpp
    */

    // Simple lightweight structures that encapsulates information about an image on device.
    // It is intended to pass to nvcc-compiled code. It is unsafe for Texture to be passed as a kernel argument, destructor will get called
// following stub destruction.

// SHOULD THE INDEX BE A FLOAT
// NEEDS TO WORK ON PRE CUDEV ARCH - NO
// CHECK IF OFFSET OBJECT CORRECTLY COPIES EVERYTHING.

namespace cv {
    namespace cudev {

        //! @addtogroup cudev
        //! @{

        template<class T, class R = T>
        struct TexturePtr {
            typedef T     elem_type, value_type;
            //typedef T     value_type;
            typedef float index_type;
            __host__ TexturePtr() {};
            __host__ TexturePtr(const cudaTextureObject_t tex_) : tex(tex_) {};
            __device__ __forceinline__ R operator ()(index_type y, index_type x) const {
                return tex2D<R>(tex, x, y);
            }
            __device__ __forceinline__ R operator ()(index_type x) const {
                //printf("Accessing : %f\n", x);
                return tex1Dfetch<R>(tex, x);
            }
        private:
            cudaTextureObject_t tex;
        };

        template<class T, class R = T>
        struct TextureOffPtr {
            typedef T     elem_type;
            typedef float index_type;
            __host__ TextureOffPtr(const cudaTextureObject_t tex_, const int yoff_, const int xoff_) : tex(tex_), yoff(yoff_), xoff(xoff_) {};
            //__host__ TextureOffPtr(const cudaTextureObject_t tex, const int yoff_, const int xoff_) : TexturePtr<T, R>(tex) {
            //    yoff = yoff_;
            //    xoff = xoff_;
            //}

            __device__ __forceinline__ R operator ()(index_type y, index_type x) const {
                return tex2D<R>(tex, x + xoff, y + yoff);
            }
        private:
            cudaTextureObject_t tex;
            int xoff = 0;
            int yoff = 0;
        };

        /** @brief non-copyable smart CUDA texture object
            *
            * UniqueTexture is a smart non-sharable wrapper for CUDA texture object handle which ensures that
            * the handle is destroyed after use.
            */
            // Return Type on operator overload
        template<class T, class R = T>
        class UniqueTexture {//}; : public TexturePtr<T, R> {
        public:
            __host__ UniqueTexture() noexcept { }
            __host__ UniqueTexture(UniqueTexture&) = delete;
            __host__ UniqueTexture(UniqueTexture&& other) noexcept {
                tex = other.tex;
                other.tex = 0;
            }

            __host__ UniqueTexture(const int rows, const int cols, T* data, const size_t step, const int nDims = 2, const bool normalizedCoords = false,
                const cudaTextureFilterMode filterMode = cudaFilterModePoint, const cudaTextureAddressMode addressMode = cudaAddressModeClamp,
                const cudaTextureReadMode readMode = cudaReadModeElementType)
            {
                create(rows, cols, data, step, nDims, normalizedCoords, filterMode, addressMode, readMode);
            }

            __host__ ~UniqueTexture() {
                printf("Destroying texture Object\n");
                if (tex != cudaTextureObject_t()) {
                    try {
                        CV_CUDEV_SAFE_CALL(cudaDestroyTextureObject(tex));
                    }
                    catch (const cv::Exception& ex) {
                        std::ostringstream os;
                        os << "Exception caught during CUDA texture object destruction.\n";
                        os << ex.what();
                        os << "Exception will be ignored.\n";
                        CV_LOG_WARNING(0, os.str().c_str());
                    }
                }

            }

            __host__ UniqueTexture& operator=(const UniqueTexture&) = delete;
            __host__ UniqueTexture& operator=(UniqueTexture&& other) noexcept {
                CV_Assert(other);
                if (&other != this) {
                    UniqueTexture(std::move(*this)); /* destroy current texture object */
                    tex = other.tex;
                    other.tex = cudaTextureObject_t();
                }
                return *this;
            }

            __host__ cudaTextureObject_t get() const noexcept {
                CV_Assert(tex);
                return tex;
            }

            __host__ explicit operator bool() const noexcept { return tex != cudaTextureect_t(); }

        private:
            __host__ void create(const int rows, const int cols, T* data, const size_t step, const int nDims, const bool normalizedCoords, const cudaTextureFilterMode filterMode,
                const cudaTextureAddressMode addressMode, const cudaTextureReadMode readMode)
            {
                CV_Assert(nDims == 1 || nDims == 2);
                //std::cout << "Constructing Texture from GlobPtrSz" << endl;
                cudaResourceDesc texRes;
                std::memset(&texRes, 0, sizeof(texRes));
                if (nDims == 1) {
                    CV_Assert(rows == 1 && cols*sizeof(T) == step);
                    texRes.resType = cudaResourceTypeLinear;
                    texRes.res.linear.devPtr = data;
                    texRes.res.linear.sizeInBytes = step;
                    texRes.res.linear.desc = cudaCreateChannelDesc<T>();
                }
                else {
                    texRes.resType = cudaResourceTypePitch2D;
                    texRes.res.pitch2D.devPtr = data;
                    texRes.res.pitch2D.height = rows;
                    texRes.res.pitch2D.width = cols;
                    texRes.res.pitch2D.pitchInBytes = step;
                    texRes.res.pitch2D.desc = cudaCreateChannelDesc<T>();
                }

                cudaTextureDesc texDescr;
                std::memset(&texDescr, 0, sizeof(texDescr));
                texDescr.normalizedCoords = normalizedCoords;
                texDescr.filterMode = filterMode;
                texDescr.addressMode[0] = addressMode;
                texDescr.addressMode[1] = addressMode;
                texDescr.addressMode[2] = addressMode;
                texDescr.readMode = readMode;

                CV_CUDEV_SAFE_CALL(cudaCreateTextureObject(&tex, &texRes, &texDescr, 0));

                printf("Constructed Unique Texture Object: %d", tex);
            }

        private:
            //typedef T     value_type;
            //typedef float index_type;
            cudaTextureObject_t tex;
        };

        /** @brief sharable smart CUDA texture object
            *
            * Texture is a smart sharable wrapper for CUDA texture handle which ensures that
            * the handle is destroyed after use.
            */
        template<class T, class R = T>
        class Texture {
        public:
            Texture(const Texture&) = default;
            Texture(Texture&&) = default;

            __host__ Texture(const int rows_, const int cols_, T* data, const size_t step, const bool normalizedCoords = false,
                const cudaTextureFilterMode filterMode = cudaFilterModePoint, const cudaTextureAddressMode addressMode = cudaAddressModeClamp,
                const cudaTextureReadMode readMode = cudaReadModeElementType, const int nDims = 2) : rows(rows_), cols(cols_),
                texture(std::make_shared<UniqueTexture<T,R>>(rows, cols, data, step, nDims, normalizedCoords, filterMode, addressMode, readMode))
            {
            }

            __host__ Texture(PtrStepSz<T> src, const bool normalizedCoords = false,
                const cudaTextureFilterMode filterMode = cudaFilterModePoint, const cudaTextureAddressMode addressMode = cudaAddressModeClamp,
                const cudaTextureReadMode readMode = cudaReadModeElementType, const int nDims = 2) :
                Texture(src.rows, src.cols, src.data, src.step, normalizedCoords, filterMode, addressMode, readMode, nDims)
            {
            }

            Texture& operator=(const Texture&) = default;
            Texture& operator=(Texture&&) = default;

            __host__ explicit operator bool() const noexcept {
                if (!texture)
                    return false;
                return texture->operator bool();
            }

            __host__ operator TexturePtr<T, R>() const
            {
                printf("Creating TexureObjPtr from Texture\n");
                return TexturePtr<T, R>(texture->get());
            }

            int rows = 0;
            int cols = 0;

        protected:
            std::shared_ptr<UniqueTexture<T, R>> texture = 0;
        };

        template <typename T, typename R> struct PtrTraits< Texture<T, R> > : PtrTraitsBase<Texture<T, R>, TexturePtr<T, R> >
        {
        };


        /** @brief sharable smart CUDA texture object
        *
        * Texture is a smart sharable wrapper for CUDA texture handle which ensures that
        * the handle is destroyed after use.
        */
        // Prevent slicing of TextureOffPtr to TexturePtr by not inheriting from Texture<T,R>
        template<class T, class R = T>
        class TextureOff { //: public Texture<T, R> {
        public:
            TextureOff(const TextureOff&) = default;
            TextureOff(TextureOff&&) = default;

            __host__ TextureOff(const int rows, const int cols, T* data, const size_t step, const int yoff_ = 0, const int xoff_ = 0, const bool normalizedCoords = false,
                const cudaTextureFilterMode filterMode = cudaFilterModePoint, const cudaTextureAddressMode addressMode = cudaAddressModeClamp,
                const cudaTextureReadMode readMode = cudaReadModeElementType, const int nDims = 2) :
                texture(std::make_shared<UniqueTexture<T, R>>(rows, cols, data, step, nDims, normalizedCoords, filterMode, addressMode, readMode)), xoff(xoff_), yoff(yoff_)
            {
            }

            __host__ TextureOff(PtrStepSz<T> src, const int yoff = 0, const int xoff = 0, const bool normalizedCoords = false,
                const cudaTextureFilterMode filterMode = cudaFilterModePoint, const cudaTextureAddressMode addressMode = cudaAddressModeClamp,
                const cudaTextureReadMode readMode = cudaReadModeElementType, const int nDims = 2) :
                TextureOff(src.rows, src.cols, src.data, src.step, yoff, xoff, normalizedCoords, filterMode, addressMode, readMode, nDims)
            {
            }

            TextureOff& operator=(const TextureOff&) = default;
            TextureOff& operator=(TextureOff&&) = default;

            __host__ operator TextureOffPtr<T, R>() const
            {
                return TextureOffPtr<T, R>(texture->get(), yoff, xoff);
            }

        private:
            int xoff = 0;
            int yoff = 0;
            std::shared_ptr<UniqueTexture<T, R>> texture = 0;
        };

 /*       template <class T> struct TextureAccessor
        {
            TextureAccessor(const PtrStepSz<T>& src) :
                tex(src, false, cudaFilterModePoint, cudaAddressModeClamp) {};

            Texture<T> tex;
            typedef T elem_type;
            typedef int index_type;

            __device__ __forceinline__ elem_type operator ()(index_type y, index_type x) const {
                return tex(y, x);
            }
        };

        template <class T> struct TextureAccessorOffset
        {
            TextureAccessorOffset(const PtrStepSz<T>& src, const int yoff_, const int xoff_) :
                tex(src, false, cudaFilterModePoint, cudaAddressModeClamp), yoff(yoff_), xoff(xoff_) {};

            Texture<T> tex;
            typedef T elem_type;
            typedef int index_type;
            int yoff;
            int xoff;

            __device__ __forceinline__ elem_type operator ()(index_type y, index_type x) const {
                return tex.get(y + yoff, x + xoff);
            }
        };*/

    }
}

#endif
