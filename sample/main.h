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

#ifndef MAIN_H
#define MAIN_H

#include <complex>
#include <vector>

#include <CL/cl.h>

#include "genFFT.h"

namespace Main
{
    // Convert float to int.
    typedef struct Converter
    {
        union Type
        {
            double f;
            int64_t i;
        };
    } Converter;

    std::vector<cl_float2> GenerateTwiddles(const int& fftSize);

    // Allocate the device buffers based on the cmd line arguments.
    cl_int CreateDeviceBuffers(cl_context context, cl_command_queue queue, const int& fftSize, const size_t& bufferSize, cl_mem& cdata, cl_mem& spectrum, cl_mem& twiddles);

    // Initialize the GPU input.
    cl_int InitializeInput(cl_command_queue queue, const int& fftSize, const std::vector<std::complex<float>>& cpuSignal, cl_mem cdata, cl_mem& twiddles);

    // Get the results.
    std::vector<cl_float2> GetOutput(cl_command_queue queue, const GenFFT& genfft, const uint32_t& size, cl_mem cdata, cl_mem spectrum, cl_int& status);

    // Generate a random signal.
    std::vector<std::complex<float>> GenerateSignal(const size_t& size);

    // Return the max of delta.real and delta.imag.
    float MaxAbsError(const std::complex<float>& ref, const cl_float2& value);

    // Compare results.
    float MaxAbsError(const std::vector<std::complex<float>>& cpuSpectrum, const std::vector<cl_float2>& gpuSpectrum);
}

#endif // !MAIN_H
