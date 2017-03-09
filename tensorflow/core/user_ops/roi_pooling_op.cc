#include "tensorflow/core/user_ops/roi_pooling_op.h"

REGISTER_OP("RoiPooling")
    .Input("feature_maps: float32")
    .Input("rois: float32")
    .Input("output_shape: int32")
    .Output("pooled_features: float32")
    .SetShapeFn([](InferenceContext* c) {

        // check inputs' rank
        ShapeHandle feature_maps_shape;
        ShapeHandle rois_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &feature_maps_shape)); // [batch, h, w, c]
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &rois_shape)); // [batch, roi, windows]

        // assign shape to output tensor
        ShapeHandle output_shape;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &output_shape)); // fixing shape here!!!
        c->set_output(0, output_shape);

        return Status::OK();
    });

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


    TensorShape result_shape = {num_batches, num_rois, output_h, output_w, num_channels};
    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, result_shape, &output_tensor));

	float* output = (float*) output_tensor->tensor_data().data();

    for(int img_i = 0; img_i < num_batches; img_i++) {
		int feature_batch_start = img_i * feature_h * feature_w * num_channels;
		int roi_batch_start = img_i * num_rois * 4;

        for(int channel_i = 0; channel_i < num_channels; channel_i++) {
            for(int roi_i = 0; roi_i < num_rois; roi_i++) {
				int roi_start = roi_batch_start + (roi_i * 4);
				int roi_h = rois[roi_start+2];
				int roi_w = rois[roi_start+3];

                // calculate kernel size
                int kernel_h = roi_h / output_h;
                int kernel_w = roi_w / output_w;

				for(int output_x = 0; output_x < output_w; output_x++) {
					for(int output_y = 0; output_y < output_y; output_y++) {
						float max_value = -1;

						for(int x = 0; x < kernel_w; x++) {
							for(int y = 0; y < kernel_h; y++) {
								int i = feature_batch_start +
											((y + (output_h * output_y)) * feature_w * num_channels) +
											((x + (output_w * output_x)) * num_channels) +
											channel_i;
								max_value = features[i] > max_value ? features[i] : max_value;
							}
						}

						output[img_i * num_rois * output_h * output_w * num_channels +
								roi_i * output_w * output_h * num_channels +
								output_y * output_w * num_channels + 
								output_x * num_channels + channel_i] = max_value;
					}
				}
			}
        }
    }
}

REGISTER_KERNEL_BUILDER(Name("RoiPooling").Device(DEVICE_CPU), RoiPoolingOp);
