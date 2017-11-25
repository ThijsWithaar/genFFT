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

#include "genFFT.h"

#include <cassert>
#include <iostream>
#include <algorithm>

using namespace std;

// --------------------------------------------------------------------------------------------------------------------------------
// Empty constructor.
// --------------------------------------------------------------------------------------------------------------------------------
GenFFT::GenFFT() :
fftSize(1),
baseFFT(32),
context(NULL),
device(NULL),
direction(Direction::FORWARD),
format(Format::C2C),
dataType(DataType::FLOAT),
simdSize(0)
{
}

// --------------------------------------------------------------------------------------------------------------------------------
// Initialization constructor.
// --------------------------------------------------------------------------------------------------------------------------------
GenFFT::GenFFT
(
const cl_context& context,
const cl_device_id& device,
const uint32_t& fftSize,
const uint32_t& baseFFT,
const Direction& direction,
const Format& format,
const DataType& dataType,
const uint32_t& simdSize
) :
fftSize(fftSize),
baseFFT(baseFFT),
factors(0),
context(context),
device(device),
direction(direction),
format(format),
dataType(dataType),
simdSize(simdSize)
{
    cl_int status = CL_SUCCESS;

    factors = Factorize(fftSize, baseFFT);

    vector<FftCore::BitRotation> rotations(factors.size());

    uint32_t stride = fftSize;
    for (size_t i = 0; i < factors.size(); i++)
    {
        FftCore stage;
        stage.globalFftSize = fftSize;

        stage.stageId.push_back((uint32_t)i);
        stage.nrStages = (uint32_t)factors.size();

        stage.fftSize.push_back(factors[i]);
        stage.simdSize = simdSize;
        stage.direction = direction;
        stage.type = format;
        stage.dataType = dataType;

        // order is important here
        rotations[i].bitRange = (int)(log(stride) / log(2));
        stride /= stage.fftSize.back();
        stage.stride.push_back(stride);
        rotations[i].steps = rotations[i].bitRange - (int)(log(stride) / log(2));

        stage.workPerWIy = workPerWIy;

        // fuse with next kernel if fusing is allowed and same FFT_SIZE
        // as a side effect localSize[1] needs to grow and can cause the workgroup size to get larger than the max workgroup size
        for (uint32_t k = 1; k < maxFuseLength; k++)
        {
            if (i == (factors.size() - 1))
                break;

            i++;
            stage.stageId.push_back((uint32_t)i);
            stage.fftSize.push_back(factors[i]);
            rotations[i].bitRange = (int)(log(stride) / log(2));
            stride /= stage.fftSize.back();
            stage.stride.push_back(stride);
            rotations[i].steps = rotations[i].bitRange - (int)(log(stride) / log(2));
        }

        if (1 == stride)
        {
            for (size_t j = 0; j < (factors.size() - 1); j++)
                stage.rotations.push_back(rotations[factors.size() - 2 - j]);
        }

        fftPipe.push_back(stage);
    }
}

// --------------------------------------------------------------------------------------------------------------------------------
// Destructor.
// --------------------------------------------------------------------------------------------------------------------------------
GenFFT::~GenFFT()
{

}

// --------------------------------------------------------------------------------------------------------------------------------
// Factor an FFT into smaller FFTs.
// --------------------------------------------------------------------------------------------------------------------------------
vector<uint32_t> GenFFT::Factorize(const uint32_t& fftSize, const uint32_t& baseFFT)
{
    vector<uint32_t> factors;

    uint32_t divider = fftSize;
    while (baseFFT <= divider)
    {
        divider /= baseFFT;
        factors.push_back(baseFFT);
    }

    if (divider != 1)
        factors.push_back(divider);

    if (2 <= factors.size())
    {
        uint32_t size1 = factors[factors.size() - 1];
        uint32_t size2 = factors[factors.size() - 2];

        bool done = false;
        if (size1 == size2)
            done = true;

        while (!done)
        {
            uint32_t crtSize1 = size1 << 1;
            uint32_t crtSize2 = size2 >> 1;

            if ((crtSize1 == size2) || (crtSize1 == crtSize2))
                done = true;

            size1 = crtSize1;
            size2 = crtSize2;
        }

        // make sure smallest FFT size is always last
        factors[factors.size() - 1] = size2;
        factors[factors.size() - 2] = size1;
    }

    return factors;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Return the FFT factors.
// --------------------------------------------------------------------------------------------------------------------------------
std::vector<uint32_t> GenFFT::Factors() const
{
    vector<uint32_t> output(factors.begin(), factors.end());

    return output;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Nr. of stages in the FFT pipeline.
// --------------------------------------------------------------------------------------------------------------------------------
size_t GenFFT::NrStages() const
{
    return factors.size();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Nr. of kernels in the FFT pipeline (after fusing some of the stages together).
// --------------------------------------------------------------------------------------------------------------------------------
size_t GenFFT::NrKernels() const
{
    return fftPipe.size();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Generate & build the code for all stages.
// --------------------------------------------------------------------------------------------------------------------------------
cl_int GenFFT::Prepare(const size_t(&globalSize)[3], const size_t(&localSize)[3])
{
    cl_int status = CL_SUCCESS;

    for (size_t i = 0; i < fftPipe.size(); i++)
    {
        fftPipe[i].globalSize[X] = globalSize[X];
        fftPipe[i].globalSize[Y] = globalSize[Y] * fftSize / fftPipe[i].fftSize[0] / fftPipe[i].workPerWIy;
        fftPipe[i].globalSize[Z] = globalSize[Z];

        fftPipe[i].localSize[X] = localSize[X];
        fftPipe[i].localSize[Y] = (1 == maxFuseLength) ? localSize[Y] : (localSize[Y] * fftSize / fftPipe[i].fftSize[0]);
        fftPipe[i].localSize[Z] = localSize[Z];

        status = fftPipe[i].SetProgram(context, device);
        if (CL_SUCCESS != status)
            return status;

        status = fftPipe[i].SetKernel();
        if (CL_SUCCESS != status)
            return status;
    }

    return status;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Set the kernel arguments for all stages.
// If only one stage the results are stored in-place (signal gets overwritten).
// Otherwise results will be stored in the second argument (spectrum).
// --------------------------------------------------------------------------------------------------------------------------------
cl_int GenFFT::SetKernelArgs(const cl_mem& rdata, const cl_mem& cdata, const cl_mem& spectrum, const cl_mem& twiddles)
{
    cl_int status = CL_SUCCESS;

    for (size_t i = 0; i < fftPipe.size(); i++)
    {
        status = fftPipe[i].SetKernelArgs(rdata, cdata, spectrum, twiddles);
        if (CL_SUCCESS != status)
            return status;
    }

    return status;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Enqueue FFT.
// --------------------------------------------------------------------------------------------------------------------------------
cl_int GenFFT::Enqueue
(
cl_command_queue queue,
cl_event* profilingEvents,
const cl_uint& num_events,
const cl_event* event_wait_list
)
{
    cl_int status = CL_SUCCESS;

    // first enqueue takes responsibility for the event wait list
    status = fftPipe[0].Enqueue(queue, (profilingEvents == NULL) ? NULL : &profilingEvents[0], num_events, event_wait_list);
    if (CL_SUCCESS != status)
        return status;

    for (size_t i = 1; i < fftPipe.size(); i++)
    {
        // if out of order capability is present
        // subsequent enqueues will have to wait for the preceding enqueue if there's an event wait list
        status = fftPipe[i].Enqueue(queue, (profilingEvents == NULL) ? NULL : &profilingEvents[i]);
        if (CL_SUCCESS != status)
            return status;
    }

    return status;
}
