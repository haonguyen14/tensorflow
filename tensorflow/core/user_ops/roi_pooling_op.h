#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace tensorflow::shape_inference;

class RoiPoolingOp : public OpKernel {
    public:
        explicit RoiPoolingOp(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext* context) override;
};

