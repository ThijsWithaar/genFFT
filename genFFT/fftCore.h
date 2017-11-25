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

#ifndef FFT_CORE_H
#define FFT_CORE_H

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>

#include <CL/cl.h>

enum ID
{
    X = 0,
    Y = 1,
    Z = 2
};

// Direction of the FFT.
enum class Direction
{
    FORWARD,
    INVERSE
};

// Input data format.
enum class Format
{
    R2C,
    C2C
};

// List of supported FFT input data types.
enum class DataType
{
    FLOAT,
    HALF
};

class GenFFT;

class FftCore
{
    friend class GenFFT;

public:
    FftCore();
    ~FftCore();

private:
    //FftCore(const FftCore& gen) = delete; // required by GenFFT: std::vector<FftCore>
    FftCore(const FftCore *const gen) = delete;
    FftCore(FftCore *const gen) = delete;
    FftCore& operator=(const FftCore& gen) = delete;
    FftCore& operator=(const FftCore *const gen) = delete;
    FftCore& operator=(FftCore *const gen) = delete;

private:
    // Bit rotation configuration.
    typedef struct BitRotation
    {
        uint32_t steps;
        uint32_t bitRange;
    } BitRotation;

private:
    // crt. stage id
    std::vector<uint32_t> stageId;
    // total nr. of stages
    uint32_t nrStages;

private:
    bool FirstStage(const uint32_t& STAGE_ID) const { return STAGE_ID == 0; }
    bool LastStage(const uint32_t& STAGE_ID) const { return STAGE_ID == (nrStages - 1); }

    bool SingleStage() const { return nrStages == 1; }
    bool MultiStage() const { return nrStages != 1; }

private:
    // global FFT size.
    uint32_t globalFftSize;
    // current FFT size.
    std::vector<uint32_t> fftSize;
    // FFT stride, up to 3D, allows chaining multiple FFTs to compute a larger FFT.
    std::vector<uint32_t> stride;
    size_t globalSize[3];
    size_t localSize[3];
    uint32_t simdSize;
    Direction direction;
    Format type;
    DataType dataType;

    // bit rotations required by the multi-kernel FFT
    std::vector<BitRotation> rotations;

    // set kernel arguments global index
    cl_uint argIndex;

    // nr. base FFTs on the Y axis to perform per workitem
    uint32_t workPerWIy;

    // source cache
    std::string source;

private:
    cl_program program;
    cl_kernel kernel;

private:
    // Get the input DataType name.
    std::string TypeNameIN() const;
    // Get the output DataType name.
    std::string TypeNameOUT() const;

private:
    // License info.
    std::string License() const;

    // Development info.
    std::string DevInfo() const;
    // Enable FP16.
    std::string EnableFP16() const;

    // Check if the SIMD value was set.
    bool SIMD_ON() const;

    // Set the SIMD attribute.
    std::string SIMD_Attribute() const;

    // Set the SIMD preprocessor constant.
    std::string SIMD_Constant() const;

    // Preprocessor defined constants.
    std::string Preprocessor() const;

    // Generate the twiddle factors.
    // We don't need to generate all of them, only (fftSize/4+1) values.
    template <class T>
    std::vector<T> TwiddleFactors(const uint32_t& fftSize) const
    {
        std::vector<T> W(fftSize / 4 + 1);

        for (uint32_t k = 0; k < fftSize / 4 + 1; k++)
            W[k] = (T)sin(2.0 * M_PI * k / fftSize);

        return W;
    }

    // Generate the twiddle factors.
    // We don't need to store all of them, only (fftSize/4+1) values.
    std::string SetTwiddleFactors() const;

    // Add function header.
    std::string FunctionHeader(const std::string& comment) const;

    // The name of the bit reversal procedure required by the Radix-2 DIF FFT algorithm.
    std::string BitReverseName(const uint32_t& fftSize) const;

    // Bit reversal procedure required by the Radix-2 DIF FFT algorithm.
    std::string BitReverse(const uint32_t& fftSize) const;

    // Bit scramble procedure required by the multi-kernel FFT algorithm.
    std::string BitUnScramble() const;

    // Multiply two complex numbers.
    std::string ComplexMul() const;

    // 2-Point Radix-2 FFT.
    std::string FFT2() const;

    // N-Point Radix-2 FFT butterflies.
    std::string Butterfly(const uint32_t& fftSize) const;

    // 1D FFT signature.
    std::string FFT1D_Signature(const uint32_t& STRIDE) const;

    // 1D FFT call.
    std::string FFT1D_Call(const uint32_t& fftSize, const uint32_t& STAGE_ID, const uint32_t& STRIDE) const;

    // 1D FFT locally defined pointers - crtrData, crtcData and crtSpectrum.
    std::string FFT1D_Pointers(const uint32_t& STRIDE) const;

    // Read the input.
    std::string FFT1D_ReadIn(const uint32_t& fftSize, const uint32_t& STAGE_ID, const uint32_t& STRIDE) const;

    // Write the output.
    std::string FFT1D_WriteOut(const uint32_t& fftSize, const uint32_t& STRIDE) const;

    // N-Point Radix-2 FFT - function name.
    std::string FFT1D_Name(const uint32_t& fftSize, const uint32_t& STAGE_ID, const uint32_t& STRIDE) const;

    // N-Point Radix-2 FFT.
    std::string FFT1D(const uint32_t& fftSize, const uint32_t& STAGE_ID, const uint32_t& STRIDE) const;

public:
    // Build the kernel name.
    std::string Kernel_Name() const;

private:
    // The signature of the kernel.
    std::string Kernel_Signature() const;

    // N-Point Radix-2 FFT kernel.
    std::string Kernel_Code() const;

    // Generate OpenCL source code.
    std::string GenerateSource() const;

public:
    // Get the OpenCL source code.
    std::string Source() const;

private:
    // Prepare the program.
    cl_int SetProgram(
        const cl_context& context,
        const cl_device_id& device
        );

    // Get program build log.
    std::string BuildLog(const cl_device_id& device, cl_int& status);

    // Prepare the kernel.
    cl_int SetKernel();

    // Set all the kernel arguments.
    cl_int SetKernelArgs(const cl_mem& rdata, const cl_mem& cdata, const cl_mem& spectrum, const cl_mem& twiddles);

public:
    // Enqueue this FFT.
    cl_int Enqueue
        (
        cl_command_queue queue,
        cl_event* profilingEvent = nullptr,
        const cl_uint& num_events = 0,
        const cl_event* event_wait_list = nullptr
        );
};

#endif // !FFT_H
