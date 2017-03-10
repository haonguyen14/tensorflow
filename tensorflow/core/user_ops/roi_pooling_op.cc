#include "tensorflow/core/user_ops/roi_pooling_op.h"
#include "tensorflow/core/platform/logging.h"

REGISTER_OP("RoiPooling")
    .Input("feature_maps: float32")
    .Input("rois: float32")
    .Input("output_shape: int32")
    .Output("pooled_features: float32")
    .Output("argmax: int32");

void RoiPoolingOp::Compute(OpKernelContext* context) {
    const Tensor& feature_tensor = context->input(0);
    const Tensor& roi_tensor = context->input(1);
    const Tensor& output_shape_tensor = context->input(2);

    OP_REQUIRES(context,
                feature_tensor.dim_size(0) == roi_tensor.dim_size(0),
                errors::InvalidArgument("mismatching batch sizes"));

    int* output_shape = (int*) output_shape_tensor.tensor_data().data(); 
    int output_h = output_shape[0];
    int output_w = output_shape[1];

    int num_batches = feature_tensor.dim_size(0);
    int num_channels = feature_tensor.dim_size(3);
    int feature_h = feature_tensor.dim_size(1);
    int feature_w = feature_tensor.dim_size(2);
    int num_rois = roi_tensor.dim_size(1);

    float* features = (float*) feature_tensor.tensor_data().data();
    float* rois = (float*) roi_tensor.tensor_data().data();

    Tensor* output_tensor;
    Tensor* argmax_tensor;
    TensorShape result_shape = {num_batches, num_rois, output_h, output_w, num_channels};
    OP_REQUIRES_OK(context, context->allocate_output(0, result_shape, &output_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, result_shape, &argmax_tensor));

	float* output = (float*) output_tensor->tensor_data().data();
    int* argmax = (int*) argmax_tensor->tensor_data().data();

    for(int img_i = 0; img_i < num_batches; img_i++) {
		int feature_batch_start = img_i * feature_h * feature_w * num_channels;
		int roi_batch_start = img_i * num_rois * 4;

        for(int channel_i = 0; channel_i < num_channels; channel_i++) {
            for(int roi_i = 0; roi_i < num_rois; roi_i++) {
				int roi_start = roi_batch_start + (roi_i * 4);

                int roi_start_y = feature_h * rois[roi_start+0];
                int roi_start_x = feature_w * rois[roi_start+1];
				int roi_h = rois[roi_start+2];
				int roi_w = rois[roi_start+3];

                // calculate kernel size
                int kernel_h = roi_h / output_h;
                int kernel_w = roi_w / output_w;

				for(int output_x = 0; output_x < output_w; output_x++) {
					for(int output_y = 0; output_y < output_h; output_y++) {
						float max_value = -1;
                        int max_index = -1;

						for(int x = 0; x < kernel_w; x++) {
							for(int y = 0; y < kernel_h; y++) {
                                int roi_x = roi_start_x + (output_x * kernel_w + x);
                                int roi_y = roi_start_y + (output_y * kernel_h + y);
                                int index = feature_batch_start +
                                            (roi_y * feature_w * num_channels) +
                                            (roi_x * num_channels) + channel_i;

                                if(features[index] > max_value) {
                                    max_value = features[index];
                                    max_index = index;
                                }
							}
						}

						int output_index = img_i * num_rois * output_h * output_w * num_channels +
								roi_i * output_w * output_h * num_channels +
								output_y * output_w * num_channels + 
								output_x * num_channels + channel_i;

                        output[output_index] = max_value;
                        argmax[output_index] = max_index;
					}
				}
			}
        }
    }
}

REGISTER_KERNEL_BUILDER(Name("RoiPooling").Device(DEVICE_CPU), RoiPoolingOp);
