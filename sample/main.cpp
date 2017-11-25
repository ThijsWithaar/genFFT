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

#include <algorithm>
#include <iostream>
#include <cstring>

#include "clutils.h"
#include "cpu.h"
#include "main.h"

using namespace std;

// --------------------------------------------------------------------------------------------------------------------------------
// 
// --------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int fftSize = 8;
    int simdSize = 8;
    int baseFFT = 32;

    // input buffer dimensions
    int cols = 1024;
    int rows = 768;

    int localSizeX = 256;

    // read the cmd line arguments if any
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-cols") == 0) { cols = atoi(argv[++i]); }
        if (strcmp(argv[i], "-rows") == 0) { rows = atoi(argv[++i]); }
        if (strcmp(argv[i], "-fft") == 0) { fftSize = atoi(argv[++i]); }
        if (strcmp(argv[i], "-base") == 0) { baseFFT = atoi(argv[++i]); }
        if (strcmp(argv[i], "-simd") == 0) { simdSize = atoi(argv[++i]); }
        if (strcmp(argv[i], "-lx") == 0) { localSizeX = atoi(argv[++i]); }
        if (strcmp(argv[i], "-h") == 0)
        {
            cout << "-cols c ... Signal width.       Default " << cols << "." << endl;
            cout << "-rows r ... Signal height.      Default " << rows << ". Must be a multiple of FFT length." << endl;
            cout << "-fft  l ... FFT length.         Default " << fftSize << "." << endl;
            cout << "-base b ... FFT base length.    Default " << baseFFT << "." << endl;
            cout << "-simd s ... Logical SIMD width. Default " << simdSize << "." << endl;
            cout << "-lx   l ... localSize[0].       Default " << localSizeX << ". Must divide the signal width." << endl;
            cout << "-h      ... Display this help." << endl;
            cout << endl;

            return 0;
        }
    }

    int bufferSize = cols * rows;
    int batchSize = bufferSize / fftSize;

    size_t localSize[3] = { static_cast<size_t>(localSizeX), 1, 1 };
    size_t globalSize[3] = { static_cast<size_t>(cols), rows / fftSize * localSize[1], 1 };
    size_t globalOffset[3] = { 0, 0, 0 };

    Direction direction = Direction::FORWARD;
    Format format = Format::C2C;
    DataType type = DataType::FLOAT;

    // 0.5s = 0.5e6us
    double interval = 0.5e6;

    int minNrRuns = 10;
    int maxNrRuns = 2000;

    // GPU execution time in us
    double minTimeGPU = 0.0;
    double maxAbsError = 0.0;

    cl_device_id gpu = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

    cl_int status = CL_SUCCESS;
    cl_mem cdata = NULL;
    cl_mem spectrum = NULL;
    cl_mem twiddles = NULL;

    if (!CLUtils::SetOclEnvironment(gpu, context, queue, deviceType))
    {
        cout << "Error setting-up an OpenCL environment." << endl << endl;

        return 1;
    }

    status = Main::CreateDeviceBuffers(context, queue, fftSize, bufferSize, cdata, spectrum, twiddles);
    if (CL_SUCCESS != status)
    {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    // run just one global/local size config
    // compute reference signal & its DFT only once and only one FFT
    vector<complex<float>> cpuData = Main::GenerateSignal(bufferSize);

    // copy the signal to the gpu
    status = Main::InitializeInput(queue, fftSize, cpuData, cdata, twiddles);

    if (CL_SUCCESS != status)
    {
        cout << "Error initializing input. CL error: " << status << endl;
        return 1;
    }

    GenFFT genfft(context, gpu, fftSize, baseFFT, direction, format, type, simdSize);

    status = genfft.Prepare(globalSize, localSize);
    if (CL_SUCCESS != status)
    {
        cout << "Error creating/building the program. CL error: " << status << endl;
        return 1;
    }

    status = genfft.SetKernelArgs(NULL, cdata, spectrum, twiddles);
    if (CL_SUCCESS != status)
    {
        cout << "Error setting the kernel arguments. CL error: " << status << endl;
        return 1;
    }

    // perform GPU FFT once for validation
    vector<cl_event> profilingEvents(genfft.NrKernels(), NULL);
    status = genfft.Enqueue(queue, &profilingEvents[0]);

    if (CL_SUCCESS != status)
    {
        cout << "Error enqueueing the kernels. CL error: " << status << endl;
        return 1;
    }

    // read the spectrum for validation
    vector<cl_float2> gpuSpectrum = Main::GetOutput(queue, genfft, bufferSize, cdata, spectrum, status);
    if (CL_SUCCESS != status)
    {
        cout << "Error getting the results. CL error: " << status << endl;
        return 1;
    }

    // get the GPU execution time
    for (size_t stage = 0; stage < genfft.NrKernels(); stage++)
    {
        minTimeGPU += CLUtils::ExecutionTime(profilingEvents[stage]);
        clReleaseEvent(profilingEvents[stage]);
    }

    // perform CPU MKL FFT once for validation; notice this is done in-place
    CPU::MKL_DFT_1D(cpuData, fftSize, rows, cols, direction == Direction::FORWARD);

    // get the error between CPU & GPU
    maxAbsError = Main::MaxAbsError(cpuData, gpuSpectrum);

    // repeat GPU FFT execution for about 100ms.
    size_t nrRuns = max(minNrRuns, min(maxNrRuns, (int)(interval / minTimeGPU)));

    vector<vector<cl_event>> events(nrRuns, vector<cl_event>(genfft.NrKernels(), NULL));
    for (size_t run = 0; run < nrRuns; run++)
        status |= genfft.Enqueue(queue, &events[run][0]);

    clFinish(queue);

    if (CL_SUCCESS != status)
    {
        cout << "Error enqueueing the kernels. CL error: " << status << endl;
        return 1;
    }

    // perform GPU FFT multiple times for performance evaluation
    vector<double> exeTimesGPU(nrRuns);
    for (size_t run = 0; run < nrRuns; run++)
    {
        for (size_t i = 0; i < genfft.NrKernels(); i++)
        {
            exeTimesGPU[run] += CLUtils::ExecutionTime(events[run][i]);
            clReleaseEvent(events[run][i]);
        }
    }
    sort(exeTimesGPU.begin(), exeTimesGPU.end());
    minTimeGPU = exeTimesGPU[0];
    minTimeGPU = (minTimeGPU < 0) ? minTimeGPU : minTimeGPU / batchSize;

    clReleaseMemObject(spectrum);
    clReleaseMemObject(cdata);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    cout << endl;
    cout << "Buffer size:            " << cols * rows << endl;
    cout << "FFT length:             " << fftSize << endl;
    cout << "Execution time per FFT: " << minTimeGPU << "us" << endl;
    cout << "Max abs error:          " << maxAbsError << endl;
    cout << endl;

    return 0;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Generate the twiddle factors used in between base FFT kernels.
// --------------------------------------------------------------------------------------------------------------------------------
vector<cl_float2> Main::GenerateTwiddles(const int& fftSize)
{
    vector<cl_float2> output(fftSize);
    for (int i = 0; i < fftSize; i++)
    {
        double angle = 2 * M_PI * i / fftSize;
        output[i].x = (float)cos(angle);
        output[i].y = (float)sin(angle);
    }

    return output;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Allocate the device buffers based on the cmd line arguments.
// --------------------------------------------------------------------------------------------------------------------------------
cl_int Main::CreateDeviceBuffers(cl_context context, cl_command_queue queue, const int& fftSize, const size_t& bufferSize, cl_mem& cdata, cl_mem& spectrum, cl_mem& twiddles)
{
    cl_int status = CL_SUCCESS;

    cdata = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(cl_float2), NULL, &status);
    if (CL_SUCCESS != status) return status;

    spectrum = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(cl_float2), NULL, &status);
    if (CL_SUCCESS != status) return status;

    twiddles = clCreateBuffer(context, CL_MEM_READ_ONLY, fftSize * sizeof(cl_float2), NULL, &status);
    if (CL_SUCCESS != status) return status;

    return status;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Initialize the GPU signal.
// --------------------------------------------------------------------------------------------------------------------------------
cl_int Main::InitializeInput(cl_command_queue queue, const int& fftSize, const vector<complex<float>>& cpuSignal, cl_mem cdata, cl_mem& twiddles)
{
    cl_int status = CL_SUCCESS;

    vector<cl_float2> w = GenerateTwiddles(fftSize);

    vector<cl_float2> cSignal(cpuSignal.size());
    for (size_t i = 0; i < cpuSignal.size(); i++)
    {
        cSignal[i].x = (float)cpuSignal[i].real();
        cSignal[i].y = (float)cpuSignal[i].imag();
    }

    status = clEnqueueWriteBuffer(queue, cdata, CL_TRUE, 0, cSignal.size() * sizeof(cl_float2), &cSignal[0], 0, NULL, NULL);

    status = clEnqueueWriteBuffer(queue, twiddles, CL_TRUE, 0, w.size() * sizeof(cl_float2), &(w)[0], 0, NULL, NULL);

    return status;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Get the results.
// --------------------------------------------------------------------------------------------------------------------------------
vector<cl_float2> Main::GetOutput(cl_command_queue queue, const GenFFT& genfft, const uint32_t& size, cl_mem cdata, cl_mem spectrum, cl_int& status)
{
    vector<cl_float2> output(size);

    cl_mem doutput = (1 == genfft.NrStages()) ? cdata : spectrum;

    status = clEnqueueReadBuffer(queue, doutput, CL_TRUE, 0, size * sizeof(cl_float2), &output[0], 0, NULL, NULL);

    return output;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Generate a random signal.
// --------------------------------------------------------------------------------------------------------------------------------
vector<complex<float>> Main::GenerateSignal(const size_t& size)
{
    vector<complex<float>> output(size);

    float vmax = 1.0f;
    float vmin = -1.0f;
    float factor = (vmax - vmin) / RAND_MAX;

    for (size_t i = 0; i < size; i++)
    {
        output[i] = { vmin + (float)std::rand() * factor, vmin + (float)std::rand() * factor };
    }

    return output;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Return the max abs error.
// TODO: are the 'isFinite' checks really necessary?
// --------------------------------------------------------------------------------------------------------------------------------
float Main::MaxAbsError(const complex<float>& ref, const cl_float2& value)
{
    float deltax = abs(ref.real() - value.x);
    float deltay = abs(ref.imag() - value.y);

    if (!isfinite(deltax))
        return numeric_limits<float>::infinity();

    if (!isfinite(deltay))
        return numeric_limits<float>::infinity();

    return max(deltax, deltay);
}

// --------------------------------------------------------------------------------------------------------------------------------
// Compare results.
// --------------------------------------------------------------------------------------------------------------------------------
float Main::MaxAbsError(const vector<complex<float>>& cpuSpectrum, const vector<cl_float2>& gpuSpectrum)
{
    float maxAbsError = -1.0;

    for (size_t i = 0; i < cpuSpectrum.size(); i++)
    {
        float absError = MaxAbsError(cpuSpectrum[i], gpuSpectrum[i]);

        if (maxAbsError < absError)
            maxAbsError = absError;
    }

    return maxAbsError;
}
