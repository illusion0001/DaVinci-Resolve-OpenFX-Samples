#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif
#include <map>
#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Blend the inputs layer by layer in the order Black Screen, p_PrevInput, p_CurrInput and p_NextInput.
const char *KernelSource = "\n" \
"__kernel void TemporalBlurKernel(int p_Width, int p_Height, float p_Blend,                                                                                                      \n" \
"                                 __global const float* p_PrevInput, __global const float* p_CurrInput, __global const float* p_NextInput, __global float* p_Output)             \n" \
"{                                                                                                                                                                               \n" \
"   const int x = get_global_id(0);                                                                                                                                              \n" \
"   const int y = get_global_id(1);                                                                                                                                              \n" \
"                                                                                                                                                                                \n" \
"   if ((x < p_Width) && (y < p_Height))                                                                                                                                         \n" \
"   {                                                                                                                                                                            \n" \
"       const int index = ((y * p_Width) + x) * 4;                                                                                                                               \n" \
"                                                                                                                                                                                \n" \
"       p_Output[index + 0] = p_Blend * (p_Blend * (1.0f - p_Blend) * p_PrevInput[index + 0] + (1 - p_Blend) * p_CurrInput[index + 0]) + (1 - p_Blend) * p_NextInput[index + 0]; \n" \
"       p_Output[index + 1] = p_Blend * (p_Blend * (1.0f - p_Blend) * p_PrevInput[index + 1] + (1 - p_Blend) * p_CurrInput[index + 1]) + (1 - p_Blend) * p_NextInput[index + 1]; \n" \
"       p_Output[index + 2] = p_Blend * (p_Blend * (1.0f - p_Blend) * p_PrevInput[index + 2] + (1 - p_Blend) * p_CurrInput[index + 2]) + (1 - p_Blend) * p_NextInput[index + 2]; \n" \
"       p_Output[index + 3] = p_Blend * (p_Blend * (1.0f - p_Blend) * p_PrevInput[index + 3] + (1 - p_Blend) * p_CurrInput[index + 3]) + (1 - p_Blend) * p_NextInput[index + 3]; \n" \
"   }                                                                                                                                                                            \n" \
"}                                                                                                                                                                               \n";

void CheckError(cl_int p_Error, const char* p_Msg)
{
    if (p_Error != CL_SUCCESS)
    {
        fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
    }
}

class Locker
{
public:
    Locker()
    {
#ifdef _WIN64
        InitializeCriticalSection(&mutex);
#else
        pthread_mutex_init(&mutex, NULL);
#endif
    }

    ~Locker()
    {
#ifdef _WIN64
        DeleteCriticalSection(&mutex);
#else
        pthread_mutex_destroy(&mutex);
#endif
    }

    void Lock()
    {
#ifdef _WIN64
        EnterCriticalSection(&mutex);
#else
        pthread_mutex_lock(&mutex);
#endif
    }

    void Unlock()
    {
#ifdef _WIN64
        LeaveCriticalSection(&mutex);
#else
        pthread_mutex_unlock(&mutex);
#endif
    }

private:
#ifdef _WIN64
    CRITICAL_SECTION mutex;
#else
    pthread_mutex_t mutex;
#endif
};

void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float p_Blend, const float* p_PrevInput, const float* p_CurrInput, const float* p_NextInput, float* p_Output)
{
    cl_int error;

    cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

    // store device id and kernel per command queue (required for multi-GPU systems)
    static std::map<cl_command_queue, cl_device_id> deviceIdMap;
    static std::map<cl_command_queue, cl_kernel> kernelMap;

    static Locker locker; // simple lock to control access to the above maps from multiple threads

    locker.Lock();

    // find the device id corresponding to the command queue
    cl_device_id deviceId = NULL;
    if (deviceIdMap.find(cmdQ) == deviceIdMap.end())
    {
        error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
        CheckError(error, "Unable to get the device");

        deviceIdMap[cmdQ] = deviceId;
    }
    else
    {
        deviceId = deviceIdMap[cmdQ];
    }

    // find the program kernel corresponding to the command queue
    cl_kernel kernel = NULL;
    if (kernelMap.find(cmdQ) == kernelMap.end())
    {
        cl_context clContext = NULL;
        error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
        CheckError(error, "Unable to get the context");

        cl_program program = clCreateProgramWithSource(clContext, 1, (const char **)&KernelSource, NULL, &error);
        CheckError(error, "Unable to create program");

        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        CheckError(error, "Unable to build program");

        kernel = clCreateKernel(program, "TemporalBlurKernel", &error);
        CheckError(error, "Unable to create kernel");

        kernelMap[cmdQ] = kernel;
    }
    else
    {
        kernel = kernelMap[cmdQ];
    }

    locker.Unlock();

    int count = 0;
    error  = clSetKernelArg(kernel, count++, sizeof(int), &p_Width);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Height);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Blend);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_PrevInput);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_CurrInput);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_NextInput);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Output);
    CheckError(error, "Unable to set kernel arguments");

    size_t localWorkSize[2], globalWorkSize[2];
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
    localWorkSize[1]  = 1;
    globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = p_Height;

    clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
