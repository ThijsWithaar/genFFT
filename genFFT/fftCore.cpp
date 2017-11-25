// Copyright (c) 2009-2016 Intel Corporation
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

#include "fftCore.h"

#include <cfloat>
#include <iomanip>
#include <vector>
#include <typeinfo>

using namespace std;

// --------------------------------------------------------------------------------------------------------------------------------
// 
// --------------------------------------------------------------------------------------------------------------------------------
FftCore::FftCore() :
    stageId(0),
    nrStages(0),
    fftSize(0),
    stride(0),
    simdSize(0),
    direction(Direction::FORWARD),
    type(Format::C2C),
    dataType(DataType::FLOAT),
    program(NULL),
    kernel(NULL),
    argIndex(0),
    workPerWIy(1)
{
    globalSize[X] = 1;
    globalSize[Y] = 1;
    globalSize[Z] = 1;
    localSize[X] = 1;
    localSize[Y] = 1;
    localSize[Z] = 1;
}

// --------------------------------------------------------------------------------------------------------------------------------
// 
// --------------------------------------------------------------------------------------------------------------------------------
FftCore::~FftCore()
{
    if (kernel == NULL)
    {
        clReleaseKernel(kernel);
        kernel = NULL;
    }

    if (program == NULL)
    {
        clReleaseProgram(program);
        program = NULL;
    }
}

// --------------------------------------------------------------------------------------------------------------------------------
// Get the input DataType name.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::TypeNameIN() const
{
    if (dataType == DataType::FLOAT)   return "float";
    if (dataType == DataType::HALF)    return "half";

    return "float";
}

// --------------------------------------------------------------------------------------------------------------------------------
// Get the output DataType name.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::TypeNameOUT() const
{
    if (dataType == DataType::FLOAT)   return "float";
    if (dataType == DataType::HALF)    return "half";

    return "float";
}

// --------------------------------------------------------------------------------------------------------------------------------
// 
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::License() const
{
    stringstream src;

    src << "// Copyright (c) 2016 Intel Corporation                                     " << endl;
    src << "// All rights reserved.                                                     " << endl;
    src << "//"                                                                           << endl;
    src << "// WARRANTY DISCLAIMER                                                      " << endl;
    src << "//"                                                                           << endl;
    src << "// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS   " << endl;
    src << "// \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT      " << endl;
    src << "// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    " << endl;
    src << "// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS      " << endl;
    src << "// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,    " << endl;
    src << "// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,      " << endl;
    src << "// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR       " << endl;
    src << "// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY      " << endl;
    src << "// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING   " << endl;
    src << "// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE      " << endl;
    src << "// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.            " << endl;
    src << "//"                                                                           << endl;
    src << "// Intel Corporation is the author of the Materials, and requests that all  " << endl;
    src << "// problem reports or change requests be submitted to it directly           " << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Development info.
// --------------------------------------------------------------------------------------------------------------------------------
std::string FftCore::DevInfo() const
{
    stringstream src;

    src << "/*/// -----------------------------------------------------------------------------------------------------------------------------" << endl;
    src << "Purpose: ";

    src << "FFT " << globalFftSize << "x1x1 ";

    src << (1 < stageId.size() ? "stages " : "stage ");
    for (size_t i = 0; i < stageId.size(); i++)
        src << stageId[i] << " ";
    src << "of " << nrStages << endl;
    src << "/*/// -----------------------------------------------------------------------------------------------------------------------------" << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Enable FP16.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::EnableFP16() const
{
    stringstream src;

    if ("half" == TypeNameOUT())
    {
        src << "#pragma OPENCL EXTENSION cl_khr_fp16         : enable" << endl;
        src << endl;
    }

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Check if the SIMD value was set.
// --------------------------------------------------------------------------------------------------------------------------------
bool FftCore::SIMD_ON() const
{
    return simdSize == 8 || simdSize == 16 || simdSize == 32;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Set the SIMD preprocessor constant.
// --------------------------------------------------------------------------------------------------------------------------------
std::string FftCore::SIMD_Constant() const
{
    std::stringstream output;

    if (SIMD_ON())
        output << "#define SIMD_SIZE     " << simdSize;

    return output.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Set the SIMD attribute.
// --------------------------------------------------------------------------------------------------------------------------------
std::string FftCore::SIMD_Attribute() const
{
    std::stringstream output;

    if (SIMD_ON())
        return "__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))";

    return output.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Preprocessor defined constants.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::Preprocessor() const
{
    stringstream src;

    src << SIMD_Constant() << endl;
    src << endl;

    src << "#define GLOBAL_SIZE_X " << globalSize[X] << endl;
    src << "#define GLOBAL_SIZE_Y " << globalSize[Y] << endl;
    src << "#define GLOBAL_SIZE_Z " << globalSize[Z] << endl;
    src << endl;

    src << "#define LOCAL_SIZE_X  " << localSize[X] << endl;
    src << "#define LOCAL_SIZE_Y  " << localSize[Y] << endl;
    src << "#define LOCAL_SIZE_Z  " << localSize[Z] << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Store the twiddle factors in the '.cl' file.
// --------------------------------------------------------------------------------------------------------------------------------
std::string FftCore::SetTwiddleFactors() const
{
    stringstream src;

    string outType = TypeNameOUT();

    vector<double> W = TwiddleFactors<double>(fftSize[0]);

    uint32_t nrTwiddles = W.size();
    nrTwiddles = nrTwiddles < 2 ? 2 : nrTwiddles;

    src << "// We don't need to store all twiddle factors:" << endl;
    src << "constant " << outType << " W[" << nrTwiddles << "] =" << endl;
    src << "{" << endl;

    int precision = DBL_DIG;
    int width = precision + 4;

    src << std::fixed;
    src << std::setprecision(precision);

    src << "    " << std::setw(width) << std::noshowpoint << std::showpos << 0.0f << "f" << "," << endl;
    for (uint32_t k = 1; k < W.size() - 1; k++)
        src << "    " << std::setw(width) << std::noshowpoint << std::showpos << W[k] << "f" << "," << endl;
    src << "    " << std::setw(width) << std::noshowpoint << std::showpos << 1.0f << "f" << "," << endl;

    src << "};" << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Add function header.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::FunctionHeader(const string& comment) const
{
    stringstream src;

    src << "// --------------------------------------------------------------------------------------------------------------------------------" << endl;
    src << "// " << comment << endl;
    src << "// --------------------------------------------------------------------------------------------------------------------------------" << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// The name of the bit reversal procedure required by the Radix-2 DIF FFT algorithm.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::BitReverseName(const uint32_t& fftSize) const
{
    stringstream src;

    src << "BitReverse_" << fftSize;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Bit reversal procedure required by the Radix-2 DIF FFT algorithm.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::BitReverse(const uint32_t& fftSize) const
{
    stringstream src;

    src << FunctionHeader("Bit reversal procedure required by the Radix-2 DIF FFT algorithm.");
    src << "int " << BitReverseName(fftSize) << "(int value)" << endl;
    src << "{" << endl;
    src << "    int reversed = 0;" << endl;
    src << endl;
    src << "    for (int j = 0; j < " << (int)(log(fftSize) / log(2)) << "; j++)" << endl;
    src << "    {" << endl;
    src << "        reversed <<= 1; reversed += value & 1; value >>= 1;" << endl;
    src << "    }" << endl;
    src << endl;
    src << "    return reversed;" << endl;
    src << "}" << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Unscrambling procedure required by the multi-kernel FFT algorithm.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::BitUnScramble() const
{
    stringstream src;

    if (0 == rotations.size())
        return src.str();

    src << FunctionHeader("Unscrambling procedure required by multi-kernel FFT algorithm.");
    src << "int BitUnScramble(int value)" << endl;
    src << "{" << endl;
    src << "    int rotated = value;" << endl;
    src << endl;

    int width = 4;
    for (size_t i = 0; i < rotations.size(); i++)
    {
        int mask = (1 << rotations[i].bitRange) - 1;

        src << "    ";
        src << "rotated = value & " << setw(width) << mask << "; ";
        src << "rotated = (value & " << setw(width + 1) << ~mask << ") | ";
        src << "((rotated << " << setw(width) << rotations[i].steps << ") & " << setw(width) << mask << ") |";
        src << "(rotated >> " << setw(width) << rotations[i].bitRange - rotations[i].steps << "); ";
        src << "value = rotated;" << endl;
    }

    src << endl;
    src << "    return rotated;" << endl;
    src << "}" << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Multiply two complex numbers.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::ComplexMul() const
{
    stringstream src;

    string outType = TypeNameOUT();

    src << FunctionHeader("Multiply two complex numbers.");

    src << "void ComplexMul(" << outType << "2* l, " << outType << "2 r)" << endl;

    src << "{" << endl;
    src << "    " << outType << "2 aux = *l;" << endl;
    src << endl;
    src << "    (*l).x = aux.x * r.x - aux.y * r.y;" << endl;
    src << "    (*l).y = aux.x * r.y + aux.y * r.x;" << endl;
    src << "}" << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// 2-Point Radix-2 FFT.
// Use temp variable or extra math, global and/or private memory.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::FFT2() const
{
    stringstream src;

    string outType = TypeNameOUT();

    src << FunctionHeader("2-Point Radix-2 FFT.");
    src << "void FFT2("<< outType << "2* x0, "<< outType << "2* x1)" << endl;

    src << "{" << endl;
    src << "    " << outType << "2 aux = *x1; *x1 = *x0 - *x1; *x0 += aux;" << endl;
    src << "}" << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// N-Point Radix-2 FFT butterflies.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::Butterfly(const uint32_t& fftSize) const
{
    stringstream src;

    string outType = TypeNameOUT();

    stringstream comment;
    comment << fftSize << "-Point Radix-2 FFT butterflies.";

    src << FunctionHeader(comment.str());

    src << "void Butterfly(" << outType << "2* x, int crtfft)" << endl;

    src << "{" << endl;
    src << "    // apply twiddle factors" << endl;
    src << "    for (int i = 0; i < crtfft / 4; i++)" << endl;
    src << "    {" << endl;
    src << "        ComplexMul(&x[2 * crtfft / 4 + i], (" << outType << "2)(+W[" << fftSize << " / 4 - " << fftSize << " / crtfft * i], -W[" << fftSize << " / crtfft * i]));" << endl;
    src << "        ComplexMul(&x[3 * crtfft / 4 + i], (" << outType << "2)(-W[" << fftSize << " / crtfft * i], -W[" << fftSize << " / 4 - " << fftSize << " / crtfft * i]));" << endl;
    src << "    }" << endl;
    src << endl;
    src << "    // perform butterflies" << endl;
    src << "    for (int i = 0; i < crtfft / 2; i++)" << endl;
    src << "        FFT2(&x[i], &x[i + crtfft / 2]);" << endl;
    src << "}" << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// 1D FFT signature.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::FFT1D_Signature(const uint32_t& STRIDE) const
{
    stringstream src;

    string inType = TypeNameIN();
    string outType = TypeNameOUT();

    switch (type)
    {
    case Format::R2C:
    {
        if (SingleStage())
            src << "(int gy, " << outType << "2* x, global " << inType << "* rdata, global " << outType << "2* cdata)" << endl;
        else
            src << "(int gy, " << outType << "2* x, global " << inType << "* rdata, global " << outType << "2* cdata, global " << outType << "2* spectrum, global " << outType << "2* twiddles)" << endl;

        break;
    }
    case Format::C2C:
    {
        if (SingleStage())
            src << "(int gy, " << outType << "2* x, global " << outType << "2* cdata)" << endl;
        else
            src << "(int gy, " << outType << "2* x, global " << outType << "2* cdata, global " << outType << "2* spectrum, global " << outType << "2* twiddles)" << endl;

        break;
    }
    }

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// 1D FFT call.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::FFT1D_Call(const uint32_t& fftSize, const uint32_t& STAGE_ID, const uint32_t& STRIDE) const
{
    stringstream src;

    string functionName = FFT1D_Name(fftSize, STAGE_ID, STRIDE);

    switch (type)
    {
    case Format::R2C:
    {
        if (SingleStage())
            src << "        " << functionName << "(gy, x, rdata, cdata);" << endl;
        else
            src << "        " << functionName << "(gy, x, rdata, cdata, spectrum, twiddles);" << endl;

        break;
    }
    case Format::C2C:
    {
        if (SingleStage())
            src << "        " << functionName << "(gy, x, cdata);" << endl;
        else
            src << "        " << functionName << "(gy, x, cdata, spectrum, twiddles);" << endl;

        break;
    }
    }

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// 1D FFT locally defined pointers - crtrData, crtcData and crtSpectrum.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::FFT1D_Pointers(const uint32_t& STRIDE) const
{
    stringstream src;

    string inType = TypeNameIN();
    string outType = TypeNameOUT();

    switch (type)
    {
    case Format::R2C:
    {
        if (SingleStage())
        {
            src << "    global " << inType << "* crtrData = rdata + crtOffset;" << endl;
            src << "    global " << outType << "2* crtcData = cdata + crtOffset;" << endl;
        }
        else
        {
            src << "    global " << inType << "* crtrData = rdata + crtOffset;" << endl;
            src << "    global " << outType << "2* crtcData = cdata + crtOffset;" << endl;
            src << "    global " << outType << "2* crtSpectrum = spectrum + crtOffset;" << endl;
        }

        break;
    }
    case Format::C2C:
    {
        if (SingleStage())
        {
            src << "    global " << outType << "2* crtcData = cdata + crtOffset;" << endl;
        }
        else
        {
            src << "    global " << outType << "2* crtcData = cdata + crtOffset;" << endl;
            src << "    global " << outType << "2* crtSpectrum = spectrum + crtOffset;" << endl;
        }

        break;
    }
    }
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Read the input.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::FFT1D_ReadIn(const uint32_t& fftSize, const uint32_t& STAGE_ID, const uint32_t& STRIDE) const
{
    stringstream src;

    string outType = TypeNameOUT();

    src << "    for (int i = 0; i < " << fftSize << "; i++)" << endl;

    if (FirstStage(STAGE_ID) && Format::R2C == type)
        src << "        x[" << BitReverseName(fftSize) << "(i)] = (" << outType << "2)((" << outType << ")crtrData[i * GLOBAL_SIZE_X * " << STRIDE << "], (" << outType << ")0.0f);" << endl;
    else
        src << "        x[" << BitReverseName(fftSize) << "(i)] = crtcData[i * GLOBAL_SIZE_X * " << STRIDE << "];" << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Write the output.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::FFT1D_WriteOut(const uint32_t& fftSize, const uint32_t& STRIDE) const
{
    stringstream src;

    // WRITE OUT
    src << "    for (int i = 0; i < " << fftSize << "; i++)" << endl;
    if (SingleStage())
        src << "        crtcData[i * GLOBAL_SIZE_X * " << STRIDE << "] = x[i];" << endl;
    else
    {
        if (1 != STRIDE)
            src << "        crtcData[i * GLOBAL_SIZE_X * " << STRIDE << "] = x[i];" << endl;
        else
            src << "        spectrum[BitUnScramble(gy * " << fftSize << " + i) * GLOBAL_SIZE_X + gx] = x[i];" << endl;
    }

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// N-Point Radix-2 FFT - function name.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::FFT1D_Name(const uint32_t& fftSize, const uint32_t& STAGE_ID, const uint32_t& STRIDE) const
{
    stringstream src;

    src << "FFT_";
    src << fftSize << "x1x1_";
    src << "stride_" << STRIDE;

    return src.str();
}


// --------------------------------------------------------------------------------------------------------------------------------
// N-Point Radix-2 FFT.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::FFT1D(const uint32_t& fftSize, const uint32_t& STAGE_ID, const uint32_t& STRIDE) const
{
    stringstream src;

    string outType = TypeNameOUT();

    stringstream comment;
    comment << fftSize << "-Point Radix-2 FFT.";

    src << FunctionHeader(comment.str());
    src << "void " << FFT1D_Name(fftSize, STAGE_ID, STRIDE);
    src << FFT1D_Signature(STRIDE);

    src << "{" << endl;

    src << "    int gx = get_global_id(0);" << endl;
    src << "    int crtOffset = (gy / " << STRIDE << ") * " << STRIDE << " * GLOBAL_SIZE_X * " << fftSize << " + (gy % " << STRIDE << ") * GLOBAL_SIZE_X + gx;" << endl;
    src << endl;

    src << FFT1D_Pointers(STRIDE);

    src << FFT1D_ReadIn(fftSize, STAGE_ID, STRIDE);

    if ((direction == Direction::INVERSE) && FirstStage(STAGE_ID))
    {
        src << "    for (int i = 0; i < " << fftSize << "; i++)" << endl;
        src << "        x[i] = (" << outType << "2)(x[i].y, x[i].x);" << endl;
    }

    src << "    for (int crtfft = 2; crtfft <= " << fftSize << "; crtfft <<= 1)" << endl;
    src << "    {" << endl;
    src << "        for (int offset = 0; offset < " << fftSize << "; offset += crtfft)" << endl;
    src << "        {" << endl;
    src << "            " << outType << "2* crtx = x + offset;" << endl;
    src << endl;
    src << "            Butterfly(crtx, crtfft);" << endl;
    src << "        }" << endl;
    src << "    }" << endl;
    src << endl;

    // apply intermediary twiddles
    if (1 != STRIDE)
    {
        // start from 1; we already know the twiddles for i=0 and gy%stride=0
        src << "    int n2 = gy % " << STRIDE << ";" << endl;
        src << "    for (int n1 = 1; n1 < " << fftSize << "; n1++)" << endl;
        src << "    {" << endl;

        src << "        int index = (" << globalFftSize << " - " << globalFftSize / (fftSize * STRIDE) << " * n1 * n2) % " << globalFftSize << ";" << endl;
        src << endl;
        src << "        ComplexMul(&x[n1], twiddles[index]);" << endl;

        src << "    }" << endl;
        src << endl;
    }

    if ((direction == Direction::INVERSE) && LastStage(STAGE_ID))
    {
        src << "    for (int i = 0; i < " << fftSize << "; i++)" << endl;
        src << "        x[i] = (" << outType << "2)(x[i].y, x[i].x);" << endl;
        src << endl;
    }

    src << FFT1D_WriteOut(fftSize, STRIDE);

    src << "}" << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Build the kernel name.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::Kernel_Name() const
{
    stringstream src;

    src << "FFT_";

    src << globalFftSize << "x1x1_";

    src << "kernel_";

    src << (1 < stageId.size() ? "stages_" : "stage_");
    for (size_t i = 0; i < stageId.size(); i++)
        src << stageId[i] << "_";
    src << "of_" << nrStages;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// The signature of the kernel.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::Kernel_Signature() const
{
    stringstream src;

    string inType = TypeNameIN();
    string outType = TypeNameOUT();

    switch (type)
    {
    case Format::R2C:
    {
        if (SingleStage())
            src << "(global " << inType << "* rdata, global " << outType << "2* cdata)" << endl;
        else
            src << "(global " << inType << "* rdata, global " << outType << "2* cdata, global " << outType << "2* spectrum, global " << outType << "2* twiddles)" << endl;

        break;
    }
    case Format::C2C:
    {
        if (SingleStage())
            src << "(global " << outType << "2* cdata)" << endl;
        else
            src << "(global " << outType << "2* cdata, global " << outType << "2* spectrum, global " << outType << "2* twiddles)" << endl;

        break;
    }
    }

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// N-Point Radix-2 FFT kernel.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::Kernel_Code() const
{
    stringstream src;

    string outType = TypeNameOUT();

    stringstream comment;
    int kernel_fftSize = 1;
    for (size_t i = 0; i < fftSize.size(); i++)
        kernel_fftSize *= fftSize[i];
    comment << kernel_fftSize << "-Point Radix-2 FFT kernel.";

    src << FunctionHeader(comment.str());

    src << "__kernel" << endl;

    src << SIMD_Attribute() << endl;

    src << "__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_Y, LOCAL_SIZE_Z)))" << endl;
    src << "void " << Kernel_Name();
    src << Kernel_Signature();

    src << "{" << endl;

    src << "    " << outType << "2 x[" << fftSize[0] << "];" << endl;
    src << endl;
    src << "    int gy = 0;" << endl;

    for (size_t i = 0; i < stride.size(); i++)
    {
        // nr. base FFTs on the Y axis to perform per workitem
        int wy = workPerWIy * fftSize[0] / fftSize[i];

        src << endl;
        src << "    for (int ky = 0; ky < " << wy << "; ky++)" << endl;
        src << "    {" << endl;
        src << "        gy = " << wy << " * get_global_id(1) + ky;" << endl;
        src << FFT1D_Call(fftSize[i], stageId[i], stride[i]);
        src << "    }" << endl;

        if ((i < (stride.size() - 1)) && (fftSize[i] * wy != globalFftSize))
        {
            src << endl;
            src << "    barrier(CLK_GLOBAL_MEM_FENCE);" << endl;
        }
    }

    src << "}" << endl;
    src << endl;

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Generate OpenCL source code.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::GenerateSource() const
{
    stringstream src;

    src << FftCore::License();

    src << FftCore::DevInfo();

    src << FftCore::EnableFP16();

    src << FftCore::Preprocessor();

    src << FftCore::SetTwiddleFactors();

    // this assumes the fftSize vector is sorted
    src << FftCore::BitReverse(fftSize[0]);
    for (size_t i = 1; i < fftSize.size(); i++)
    {
        if (fftSize[i] == fftSize[i - 1])
            continue;

        src << FftCore::BitReverse(fftSize[i]);
    }

    src << FftCore::BitUnScramble();

    src << FftCore::ComplexMul();

    src << FftCore::FFT2();

    src << FftCore::Butterfly(fftSize[0]);

    for (size_t i = 0; i < stride.size(); i++)
        src << FftCore::FFT1D(fftSize[i], stageId[i], stride[i]);

    src << FftCore::Kernel_Code();

    return src.str();
}

// --------------------------------------------------------------------------------------------------------------------------------
// Get the OpenCL source code.
// --------------------------------------------------------------------------------------------------------------------------------
string FftCore::Source() const
{
    return source;
}

// ------------------------------------------------------------------------------------------------
// Get program build log.
// ------------------------------------------------------------------------------------------------
string FftCore::BuildLog(const cl_device_id& device, cl_int& status)
{
    cl_program_build_info info = CL_PROGRAM_BUILD_LOG;
    size_t size = 0;

    status = clGetProgramBuildInfo(program, device, info, 0, NULL, &size);
    if (CL_SUCCESS != status)
        return "";

    // last info char is '\0' so allocate a size-1 string
    string output(size - 1, '\0');
    status = clGetProgramBuildInfo(program, device, info, size, &output[0], NULL);
    if (CL_SUCCESS != status)
        return "";

    return output;
}

// ------------------------------------------------------------------------------------------------
// Prepare the program.
// ------------------------------------------------------------------------------------------------
cl_int FftCore::SetProgram
(
const cl_context& context,
const cl_device_id& device
)
{
    cl_int status = CL_SUCCESS;

    source = GenerateSource();

    char* buffer = (char*)source.c_str();
    size_t length = source.length();

    program = clCreateProgramWithSource(context, 1, const_cast<const char**>(&buffer), &length, &status);
    if (CL_SUCCESS != status)
        return status;

    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    return status;
}

// ------------------------------------------------------------------------------------------------
// Prepare the kernel.
// ------------------------------------------------------------------------------------------------
cl_int FftCore::SetKernel()
{
    cl_int status = CL_SUCCESS;

    kernel = clCreateKernel(program, Kernel_Name().c_str(), &status);

    return status;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Set all the kernel arguments.
// --------------------------------------------------------------------------------------------------------------------------------
cl_int FftCore::SetKernelArgs(const cl_mem& rdata, const cl_mem& cdata, const cl_mem& spectrum, const cl_mem& twiddles)
{
    cl_int status = CL_SUCCESS;

    switch (type)
    {
    case Format::R2C:
    {
        if (SingleStage())
        {
            status = clSetKernelArg(kernel, (cl_uint)argIndex++, sizeof(cl_mem), &rdata);
            if (CL_SUCCESS != status) return status;

            status = clSetKernelArg(kernel, (cl_uint)argIndex++, sizeof(cl_mem), &cdata);
            if (CL_SUCCESS != status) return status;
        }
        else
        {
            status = clSetKernelArg(kernel, (cl_uint)argIndex++, sizeof(cl_mem), &rdata);
            if (CL_SUCCESS != status) return status;

            status = clSetKernelArg(kernel, (cl_uint)argIndex++, sizeof(cl_mem), &cdata);
            if (CL_SUCCESS != status) return status;

            status = clSetKernelArg(kernel, (cl_uint)argIndex++, sizeof(cl_mem), &spectrum);
            if (CL_SUCCESS != status) return status;

            status = clSetKernelArg(kernel, (cl_uint)argIndex++, sizeof(cl_mem), &twiddles);
            if (CL_SUCCESS != status) return status;
        }

        break;
    }
    case Format::C2C:
    {
        if (SingleStage())
        {
            status = clSetKernelArg(kernel, (cl_uint)argIndex++, sizeof(cl_mem), &cdata);
            if (CL_SUCCESS != status) return status;
        }
        else
        {
            status = clSetKernelArg(kernel, (cl_uint)argIndex++, sizeof(cl_mem), &cdata);
            if (CL_SUCCESS != status) return status;

            status = clSetKernelArg(kernel, (cl_uint)argIndex++, sizeof(cl_mem), &spectrum);
            if (CL_SUCCESS != status) return status;

            status = clSetKernelArg(kernel, (cl_uint)argIndex++, sizeof(cl_mem), &twiddles);
            if (CL_SUCCESS != status) return status;
        }

        break;
    }
    }

    return status;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Enqueue this FFT.
// --------------------------------------------------------------------------------------------------------------------------------
cl_int FftCore::Enqueue
(
cl_command_queue queue,
cl_event* profilingEvent,
const cl_uint& num_events,
const cl_event* event_wait_list
)
{
    return clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalSize, localSize, num_events, event_wait_list, profilingEvent);
}
