// Copyright (c) 2016 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly

#ifndef GEN_FFT_H
#define GEN_FFT_H

#include "fftCore.h"

#include <vector>

class GenFFT
{
public:
    GenFFT();
    GenFFT(const cl_context& context,
        const cl_device_id& device,
        const uint32_t& fftSize,
        const uint32_t& baseFFT,
        const Direction& direction,
        const Format& type,
        const DataType& dataType,
        const uint32_t& simdSize);
    ~GenFFT();

private:
    GenFFT(const GenFFT& gen) = delete;
    GenFFT(const GenFFT *const gen) = delete;
    GenFFT(GenFFT *const gen) = delete;
    GenFFT& operator=(const GenFFT& gen) = delete;
    GenFFT& operator=(const GenFFT *const gen) = delete;
    GenFFT& operator=(GenFFT *const gen) = delete;

private: // CL environment
    cl_context context;
    cl_device_id device;

private:
    // This is the size for which best performance is achieved.
    // The FFT will be factored into FFTs smaller than this optimum FFT size.
    uint32_t baseFFT;
    // FFT size.
    uint32_t fftSize;

private:
    std::vector<uint32_t> factors;
    std::vector<FftCore> fftPipe;
    const Direction direction;
    const Format format;
    const DataType dataType;
    uint32_t simdSize;

private:
    // max nr. of same FFT_SIZE kernels to fuse
    const uint32_t maxFuseLength = 1;
    // nr. base FFTs on the Y axis to perform per workitem
    const uint32_t workPerWIy = 1;

private:
    // Factor an FFT into smaller FFTs.
    static std::vector<uint32_t> Factorize(const uint32_t& fftSize, const uint32_t& baseFFT);

public:
    // The const indexer.
    const FftCore& operator[](const size_t& stage) const { return fftPipe[stage]; };
    // The indexer.
    FftCore& operator[](const size_t& stage) { return fftPipe[stage]; };

public:
    // Return the FFT factors.
    std::vector<uint32_t> Factors() const;
    // Nr. of stages in the FFT pipeline.
    size_t NrStages() const;
    // Nr. of kernels in the FFT pipeline (after fusing some of the stages together).
    size_t NrKernels() const;

public:
    // Generate & builds the code for all stages.
    cl_int Prepare(const size_t(&globalSize)[3], const size_t(&localSize)[3]);

    // Set the kernel arguments for all stages.
    // If only one stage the results are stored in-place (signal gets overwritten).
    // Otherwise results will be stored in the second argument (spectrum).
    cl_int SetKernelArgs(const cl_mem& rdata, const cl_mem& cdata, const cl_mem& spectrum, const cl_mem& twiddles);

    // Enqueue FFT.
    cl_int Enqueue
        (
        cl_command_queue queue,
        cl_event* profilingEvents = nullptr,
        const cl_uint& num_events = 0,
        const cl_event* event_wait_list = nullptr
        );
};

#endif // !GEN_H
