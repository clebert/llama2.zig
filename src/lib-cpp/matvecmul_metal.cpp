#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include "QuartzCore/QuartzCore.hpp"
#include <Metal/Metal.hpp>
#include "matvecmul_metal.h"

extern "C"
{
  void matvecmulMetal(
      const float *row_major_matrix,
      const float *input_vector,
      float *output_vector,
      u_int64_t m_rows,
      u_int64_t n_cols)
  {
    NS::AutoreleasePool *pAutoreleasePool = NS::AutoreleasePool::alloc()->init();

    MTL::Device *pDevice = MTL::CreateSystemDefaultDevice();

    pDevice->autorelease();

    NS::Error *pError = nullptr;

    const char *pSource = R"(
      #include <metal_stdlib>
      using namespace metal;

      kernel void matvecmul(
        const device float* row_major_matrix [[ buffer(0) ]],
        const device float* input_vector [[ buffer(1) ]],
        device float* output_vector [[ buffer(2) ]],
        const device uint* n_cols [[ buffer(3) ]],
        uint gid [[ thread_position_in_grid ]]) {

        float sum = 0.0;

        for (uint col = 0; col < *n_cols; ++col) {
          sum += row_major_matrix[gid * *n_cols + col] * input_vector[col];
        }

        output_vector[gid] = sum;
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
        pLibrary->newFunction(NS::String::string("matvecmul", NS::UTF8StringEncoding));

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

    MTL::Buffer *pMatrixBuffer =
        pDevice->newBuffer(m_rows * n_cols * sizeof(float), MTL::ResourceStorageModeShared);

    pMatrixBuffer->autorelease();

    MTL::Buffer *pInputVectorBuffer =
        pDevice->newBuffer(n_cols * sizeof(float), MTL::ResourceStorageModeShared);

    pInputVectorBuffer->autorelease();

    MTL::Buffer *pOutputVectorBuffer =
        pDevice->newBuffer(m_rows * sizeof(float), MTL::ResourceStorageModeShared);

    pOutputVectorBuffer->autorelease();

    memcpy(pMatrixBuffer->contents(), row_major_matrix, m_rows * n_cols * sizeof(float));
    memcpy(pInputVectorBuffer->contents(), input_vector, n_cols * sizeof(float));

    pCommandEncoder->setBuffer(pMatrixBuffer, 0, 0);
    pCommandEncoder->setBuffer(pInputVectorBuffer, 0, 1);
    pCommandEncoder->setBuffer(pOutputVectorBuffer, 0, 2);
    pCommandEncoder->setBytes(&n_cols, sizeof(size_t), 3);

    MTL::Size threadsPerGrid = MTL::Size(m_rows, 1, 1);
    MTL::Size threadsPerThreadgroup = MTL::Size(pPipelineState->threadExecutionWidth(), 1, 1);

    pCommandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
    pCommandEncoder->endEncoding();

    pCommandBuffer->commit();
    pCommandBuffer->waitUntilCompleted();

    memcpy(output_vector, pOutputVectorBuffer->contents(), m_rows * sizeof(float));

    pAutoreleasePool->release();
  }
}
