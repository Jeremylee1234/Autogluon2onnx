import onnx 


def mean_operator(ouput_name,inputs_name,num_of_childs):
    mean = onnx.helper.make_node(
                        "Mean",
                        inputs=[inputs_name+ str(i) for i in range(num_of_childs)],
                        outputs=[ouput_name])
    return mean

def argmax_operator(ouput_name,inputs_name):
    argmax = onnx.helper.make_node(
                        "ArgMax", inputs=[inputs_name], outputs=[ouput_name], axis=1, keepdims=0)
    return argmax

def softmax_operator(ouput_name,inputs_name):
    softmax = onnx.helper.make_node(
                        "Softmax", inputs=[inputs_name], outputs=[ouput_name])
    return softmax