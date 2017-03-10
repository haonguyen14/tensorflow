#include "tensorflow/core/user_ops/roi_pooling_grad_op.h"

REGISTER_OP("RoiPoolingGrad")
    .Input("input_shape: int32")
    .Input("gradients: float32")
    .Input("argmax: int32")
    .Output("output: float32");

void RoiPoolingGradOp::Compute(OpKernelContext* context) {
    const Tensor& input_shape_tensor = context->input(0);
    const Tensor& gradient_tensor = context->input(1);
    const Tensor& argmax_tensor = context->input(2);

    int* input_shape = (int*) input_shape_tensor.tensor_data().data();
    int num_batches = input_shape[0];
    int feature_height = input_shape[1];
    int feature_width = input_shape[2];
    int num_channels = input_shape[3];

    int* argmax = (int*) argmax_tensor.tensor_data().data();
    int num_rois = argmax_tensor.dim_size(1);
    int argmax_height = argmax_tensor.dim_size(2);
    int argmax_width = argmax_tensor.dim_size(3);

    float* gradients = (float*) gradient_tensor.tensor_data().data();
    
    TensorShape output_shape = {
        num_batches, feature_height, feature_width, num_channels};

    Tensor* output_tensor;
    OP_REQUIRES_OK(context,
            context->allocate_output(0, output_shape, &output_tensor));

    float* output = (float*) output_tensor->tensor_data().data();
	int output_size = num_batches * feature_height * feature_width * num_channels;
	for(int i = 0; i < output_size; i++) output[i] = 0.0;

	int argmax_size = num_batches * num_rois * argmax_height * argmax_width * num_channels; 
    for(int i = 0; i < argmax_size; i++) {
		output[argmax[i]] += gradients[i];
    }
}

REGISTER_KERNEL_BUILDER(Name("RoiPoolingGrad")
        .Device(DEVICE_CPU), RoiPoolingGradOp);
