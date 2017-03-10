#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class RoiPoolingGradOp : public OpKernel {
    public:
        explicit RoiPoolingGradOp(OpKernelConstruction* context)
                    : OpKernel(context) {}

        void Compute(OpKernelContext* context) override;
};
