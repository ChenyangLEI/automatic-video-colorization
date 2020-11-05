import tensorflow as tf

_flow_warp_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("./lib/flow_warp.so"))


def flow_warp(image, flow):
    """
    Return the flow flow for a flow.

    Args:
        image: (array): write your description
        flow: (todo): write your description
    """
    return _flow_warp_ops.flow_warp(image, flow)


@tf.RegisterGradient("FlowWarp")
def _flow_warp_grad(flow_warp_op, gradients):
    """
    Returns the gradient_warp_warp.

    Args:
        flow_warp_op: (bool): write your description
        gradients: (todo): write your description
    """
    return _flow_warp_ops.flow_warp_grad(flow_warp_op.inputs[0],
                                         flow_warp_op.inputs[1],
                                         gradients)
