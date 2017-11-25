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

#include "clutils.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>

//#include "IO.h"

using namespace std;

// --------------------------------------------------------------------------------------------------------------------------------
// Get the platforms available on this machine.
// --------------------------------------------------------------------------------------------------------------------------------
vector<cl_platform_id> CLUtils::GetPlatforms()
{
    cl_int status = CL_SUCCESS;

    cl_uint size = 0;
    status = clGetPlatformIDs(0, NULL, &size);
    if (CL_SUCCESS != status)
        return vector<cl_platform_id>();

    vector<cl_platform_id> output(size);
    status = clGetPlatformIDs(size, &output[0], NULL);
    if (CL_SUCCESS != status)
        return vector<cl_platform_id>();

    return output;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Get the specified platform info.
// --------------------------------------------------------------------------------------------------------------------------------
string CLUtils::Get(const cl_platform_id& platform, const cl_platform_info& info)
{
    cl_int status = CL_SUCCESS;
    size_t size = 0;

    status = clGetPlatformInfo(platform, info, 0, NULL, &size);
    if (CL_SUCCESS != status)
        return new char[0];

    char* chars = new char[size];
    memset(chars, 0, size);

    status = clGetPlatformInfo(platform, info, size, chars, NULL);
    if (CL_SUCCESS != status)
    {
        delete[] chars;
        return new char[0];
    }

    if (chars == NULL)
        return "";

    string buffer = chars;

    delete[] chars;

    return buffer;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Get the platform for a specific vendor available on this machine.
// --------------------------------------------------------------------------------------------------------------------------------
cl_platform_id CLUtils::GetPlatform(const string& vendorName)
{
    cl_platform_id platform = NULL;

    string vendorLowerCase = vendorName;
    transform(vendorLowerCase.begin(), vendorLowerCase.end(), vendorLowerCase.begin(), ::tolower);

    vector<cl_platform_id> platforms = GetPlatforms();
    for (size_t i = 0; i < platforms.size(); i++)
    {
        string vendorInfo = Get(platforms[i], CL_PLATFORM_VENDOR);
        string nameInfo = Get(platforms[i], CL_PLATFORM_NAME);

        transform(vendorInfo.begin(), vendorInfo.end(), vendorInfo.begin(), ::tolower);
        transform(nameInfo.begin(), nameInfo.end(), nameInfo.begin(), ::tolower);

        bool match = (vendorInfo.npos != vendorInfo.find(vendorLowerCase)) || (nameInfo.npos != nameInfo.find(vendorName));
        bool isNotExperimental = nameInfo.npos == nameInfo.find("experimental");

        if (match && isNotExperimental)
        {
            platform = platforms[i];
            return platform;
        }
    }

    return platform;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Get the devices available in the specified platform.
// --------------------------------------------------------------------------------------------------------------------------------
vector<cl_device_id> CLUtils::GetDevices(const cl_platform_id& platform, const cl_device_type& type)
{
    vector<cl_device_id> output;

    cl_int status = CL_SUCCESS;

    cl_uint size = 0;
    status = clGetDeviceIDs(platform, type, 0, NULL, &size);
    if (CL_SUCCESS != status)
        return output;

    cl_device_id* devices = new cl_device_id[size];

    status = clGetDeviceIDs(platform, type, size, devices, NULL);
    if (CL_SUCCESS != status)
        return output;

    output = vector<cl_device_id>(devices, devices + size);

    delete[] devices;

    return output;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Get context & queue.
// Prepare the OpenCL tools.
// --------------------------------------------------------------------------------------------------------------------------------
bool CLUtils::SetOclEnvironment
(
    cl_device_id& device,
    cl_context &context,
    cl_command_queue &queue,
    const cl_device_type& type
)
{
    cl_int status = CL_SUCCESS;

    device = NULL;

    cl_platform_id platform = CLUtils::GetPlatform("Intel");
    if (NULL == platform)
        return false;

    vector<cl_device_id> devices = CLUtils::GetDevices(platform, type);
    if (devices.size() < 1)
        return false;

    device = devices[0];

    cout << "Device name:             " << CLUtils::Get<string>(device, CL_DEVICE_NAME) << endl;
    cout << "Device type:             ";
    switch (type)
    {
    case CL_DEVICE_TYPE_ACCELERATOR: cout << "ACCELERATOR" << endl; break;
    case CL_DEVICE_TYPE_ALL:         cout << "ALL        " << endl; break;
    case CL_DEVICE_TYPE_CPU:         cout << "CPU        " << endl; break;
    case CL_DEVICE_TYPE_CUSTOM:      cout << "CUSTOM     " << endl; break;
    case CL_DEVICE_TYPE_DEFAULT:     cout << "DEFAULT    " << endl; break;
    case CL_DEVICE_TYPE_GPU:         cout << "GPU        " << endl; break;
    }
    cout << "Device version:          " << CLUtils::Get<string>(device, CL_DEVICE_VERSION) << endl;
    cout << "Device OpenCL C version: " << CLUtils::Get<string>(device, CL_DEVICE_OPENCL_C_VERSION) << endl;
    cout << "Driver version:          " << CLUtils::Get<string>(device, CL_DRIVER_VERSION) << endl;
    cout << endl;

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    if (CL_SUCCESS != status)
        return false;

    cl_queue_properties qProperties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, qProperties, &status);
    if (CL_SUCCESS != status)
        return false;

    return true;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Get the execution time in us.
// Make sure to call 'clFinish' before this.
// --------------------------------------------------------------------------------------------------------------------------------
double CLUtils::ExecutionTime(const cl_event& event)
{
    double output = numeric_limits<double>::quiet_NaN();

    // conversion factor to us
    const double us = 1e-3;

    if (event == NULL)
        return output;

    cl_int status = CL_SUCCESS;

    cl_ulong start = 0;
    status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    if (CL_SUCCESS != status)
        return output;

    cl_ulong end = 0;
    status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    if (CL_SUCCESS != status)
        return output;

    output = (end - start) * us;

    return output;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Get device info (string specialization).
// --------------------------------------------------------------------------------------------------------------------------------
template<> string CLUtils::Get<string>(const cl_device_id& device, const cl_device_info& info)
{
    cl_int status = CL_SUCCESS;
    size_t size = 0;

    status = clGetDeviceInfo(device, info, 0, NULL, &size);
    if (CL_SUCCESS != status)
        return "";

    // last info char is '\0' so allocate a size-1 string
    string output(size - 1, '\0');
    status = clGetDeviceInfo(device, info, size, &output[0], NULL);
    if (CL_SUCCESS != status)
        return "";

    return output;
}
