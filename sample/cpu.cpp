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

#include "cpu.h"

#define _USE_MATH_DEFINES
#include <math.h>

#ifdef USE_MKL
	#include "mkl_dfti.h"
#endif

using namespace std;

// ------------------------------------------------------------------------------------------------
// CPU MKL 1D DFT in-place implementation. Column major only.
// ------------------------------------------------------------------------------------------------
void CPU::MKL_DFT_1D(vector<complex<float>>& data, const size_t& fftSize, const size_t& rows, const size_t& cols, const bool& forward)
{
#ifdef USE_MKL
    size_t length = data.size();

    DFTI_DESCRIPTOR_HANDLE mklFFT;

    MKL_LONG status;

    status = DftiCreateDescriptor(&mklFFT, DFTI_SINGLE, DFTI_COMPLEX, 1, fftSize);

    status = DftiSetValue(mklFFT, DFTI_NUMBER_OF_TRANSFORMS, cols);
    status = DftiSetValue(mklFFT, DFTI_INPUT_DISTANCE, 1);
    status = DftiSetValue(mklFFT, DFTI_OUTPUT_DISTANCE, 1);

    MKL_LONG stride[2] = { 0, (long)cols };
    status = DftiSetValue(mklFFT, DFTI_INPUT_STRIDES, stride);
    status = DftiSetValue(mklFFT, DFTI_OUTPUT_STRIDES, stride);

    status = DftiCommitDescriptor(mklFFT);

    for (size_t row = 0; row < rows / fftSize; row++)
    {
        size_t offset = row * cols * fftSize;

        switch (forward)
        {
        case true:  status = DftiComputeForward(mklFFT, &data[offset]);  break;
        case false: status = DftiComputeBackward(mklFFT, &data[offset]); break;
        }
    }

    status = DftiFreeDescriptor(&mklFFT);
#endif
}
