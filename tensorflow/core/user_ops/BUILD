load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name="roi_pooling.so",
    srcs=["roi_pooling_op.h", "roi_pooling_op.cc"],
    deps=[":roi_pooling_grad.so"]
)

tf_custom_op_library(
    name="roi_pooling_grad.so",
    srcs=["roi_pooling_grad_op.h", "roi_pooling_grad_op.cc"]
)
