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

#ifndef CLUTILS_H
#define CLUTILS_H

#include <CL/opencl.h>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <string>
#include <vector>

typedef unsigned char uchar;

namespace CLUtils
{
    // Get the platforms available on this machine.
    std::vector<cl_platform_id> GetPlatforms();

    // Get the specified platform info.
    std::string Get(const cl_platform_id& platform, const cl_platform_info& info);

    // Get the platform for a specific vendor available on this machine.
    cl_platform_id GetPlatform(const std::string& vendorName);

    // Get the devices available in the specified platform.
    std::vector<cl_device_id> GetDevices(const cl_platform_id& platform, const cl_device_type& type = CL_DEVICE_TYPE_ALL);

    // Get context & queue.
    bool SetOclEnvironment
        (
        cl_device_id& device,
        cl_context &context,
        cl_command_queue &queue,
        const cl_device_type& type = CL_DEVICE_TYPE_GPU
        );

    // Get the execution time in us.
    // Make sure to call 'clFinish' before this.
    double ExecutionTime(const cl_event& event);

    // Get device info.
    template<class T> T Get(const cl_device_id& device, const cl_device_info& info)
    {
        cl_int status = CL_SUCCESS;
        size_t size = 0;

        status = clGetDeviceInfo(device, info, 0, NULL, &size);

        if (CL_SUCCESS != status)
            return (T)0;

        T output = 0;

        status = clGetDeviceInfo(device, info, size, &output, NULL);

        if (CL_SUCCESS != status)
            return (T)0;

        return output;
    }

    // Get device info (std::string specialization).
    template<> std::string Get<std::string>(const cl_device_id& device, const cl_device_info& info);
}

#endif
