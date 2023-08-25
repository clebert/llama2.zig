#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include "QuartzCore/QuartzCore.hpp"
#include <Metal/Metal.hpp>
#include "matmul_metal.h"

extern "C"
{
  void matmulMetal(float *result, const float *a, const float *b, size_t result_len, size_t a_len)
  {
    NS::AutoreleasePool *pAutoreleasePool = NS::AutoreleasePool::alloc()->init();

    MTL::Device *pDevice = MTL::CreateSystemDefaultDevice();

    pDevice->autorelease();

    NS::Error *pError = nullptr;

    const char *pSource = R"(
      #include <metal_stdlib>
      using namespace metal;

      kernel void matmul(
        device float* result [[ buffer(0) ]],
        const device float* a [[ buffer(1) ]],
        const device float* b [[ buffer(2) ]],
        const device uint* a_len [[ buffer(3) ]],
        uint gid [[ thread_position_in_grid ]]) {

        float sum = 0.0;

        for (uint k = 0; k < *a_len; ++k) {
          sum += a[k] * b[gid * *a_len + k];
        }

        result[gid] = sum;
      }
    )";

    MTL::Library *pLibrary = pDevice->newLibrary(
        NS::String::string(pSource, NS::UTF8StringEncoding), nullptr, &pError);

    if (!pLibrary)
    {
      __builtin_printf("%s", pError->localizedDescription()->utf8String());
      assert(false);
    }

    pLibrary->autorelease();

    MTL::Function *pFunction =
        pLibrary->newFunction(NS::String::string("matmul", NS::UTF8StringEncoding));

    if (!pFunction)
    {
      __builtin_printf("%s", pError->localizedDescription()->utf8String());
      assert(false);
    }

    pFunction->autorelease();

    MTL::ComputePipelineState *pPipelineState =
        pDevice->newComputePipelineState(pFunction, &pError);

    if (!pPipelineState)
    {
      __builtin_printf("%s", pError->localizedDescription()->utf8String());
      assert(false);
    }

    pPipelineState->autorelease();

    MTL::CommandQueue *pCommandQueue = pDevice->newCommandQueue();

    pCommandQueue->autorelease();

    MTL::CommandBuffer *pCommandBuffer = pCommandQueue->commandBuffer();
    MTL::ComputeCommandEncoder *pCommandEncoder = pCommandBuffer->computeCommandEncoder();

    pCommandEncoder->setComputePipelineState(pPipelineState);

    MTL::Buffer *pResultBuffer =
        pDevice->newBuffer(result_len * sizeof(float), MTL::ResourceStorageModeShared);

    pResultBuffer->autorelease();

    MTL::Buffer *pABuffer =
        pDevice->newBuffer(a_len * sizeof(float), MTL::ResourceStorageModeShared);

    pABuffer->autorelease();

    MTL::Buffer *pBBuffer =
        pDevice->newBuffer(a_len * result_len * sizeof(float), MTL::ResourceStorageModeShared);

    pBBuffer->autorelease();

    memcpy(pABuffer->contents(), a, a_len * sizeof(float));
    memcpy(pBBuffer->contents(), b, a_len * result_len * sizeof(float));

    pCommandEncoder->setBuffer(pResultBuffer, 0, 0);
    pCommandEncoder->setBuffer(pABuffer, 0, 1);
    pCommandEncoder->setBuffer(pBBuffer, 0, 2);
    pCommandEncoder->setBytes(&a_len, sizeof(size_t), 3);

    MTL::Size threadsPerGrid = MTL::Size(result_len, 1, 1);
    MTL::Size threadsPerThreadgroup = MTL::Size(pPipelineState->threadExecutionWidth(), 1, 1);

    pCommandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
    pCommandEncoder->endEncoding();

    pCommandBuffer->commit();
    pCommandBuffer->waitUntilCompleted();

    memcpy(result, pResultBuffer->contents(), result_len * sizeof(float));

    pAutoreleasePool->release();
  }
}
