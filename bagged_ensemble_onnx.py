
import argparse
import os
import pickle
from os.path import isdir, isfile, join

import lightgbm as lgb
import numpy as np
import onnx
import onnxmltools
import onnxruntime as ort
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.tabular_nn.torch.torch_network_modules import \
    EmbedNet
from lightgbm.basic import Booster as lgbmBooster
from onnxconverter_common.data_types import (DictionaryType, DoubleTensorType,
                                             FloatTensorType, Int64TensorType)
from skl2onnx import convert_sklearn

from operators import argmax_operator, mean_operator, softmax_operator
from utils import model_dir_tools


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",type=str)
    args = parser.parse_args()
    return args


class bagged_model2onnx():

    """
    This class is used to load the bagged model and convert it to onnx format.
    Warning: The outputs of LightGBM onnx model are not always as same as the original model.
    Parameters
    ----------
    model_path : str
        Path to the bagged_model in autogulon directory. e.g. 'AutogluonModels/ag_model_2021-05-20_15-00-00/models/LightGBMLarge'
    """
    def __init__(self,model_dir) -> None:
        #To initiate this 
        self.bagged_model_path, self.childs_path = model_dir_tools(model_dir)
        self.bagged_model = self.load_model(self.bagged_model_path)
        self.childs = [self.bagged_model.load_child(child) for child in self.bagged_model.models]
        self.input_format = list(self.bagged_model.feature_metadata.get_type_group_map_raw().items())
        #TODO: initial_types supposed to be extract from the bagged_model pkl file, 
        # but some models like NeuralTorch which predict method processed sklearn ColumnTransformer output, the initial_types is not the same as the original model.
        self.initial_types = []
        for i in range(len(self.input_format)):
            if self.input_format[i][0] == 'int':
                self.initial_types.append(('input'+str(i),Int64TensorType([-1,len(self.input_format[i][1])])))
            elif self.input_format[i][0] == 'float':
                self.initial_types.append(('input'+str(i),FloatTensorType([-1,len(self.input_format[i][1])])))

    def load_model(self,model_path):
        """
        This function is used to load models.
        """
        if isinstance(model_path, str):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif isinstance(model_path, list):
            model_list = []
            for dir in list:
                with open(dir, 'rb') as f:
                    model_list.append(pickle.load(f)) 
            return model_list
    
    def child_to_onnx(self):
        """
        This function is used to convert child models to onnx format.
        """
        if self.childs:
            if isinstance(self.childs[0].model,lgbmBooster):
                self.child_type = lgbmBooster
                self.onnx_models = [onnxmltools.convert_lightgbm(model.model, initial_types= self.initial_types,target_opset=15,zipmap=False) for model in self.childs]
            # elif isinstance(self.childs[0],catboost.CatBoostClassifier):
            # elif isinstance(self.childs[0].model,EmbedNet):
            #     self.child_type = EmbedNet
                
    def save_child_onnx(self,batch_size = None):
        """
        This function is used to save the onnx models.
        """
        if self.childs:
            if isinstance(self.childs[0].model,lgbmBooster):
                self.child_type = lgbmBooster
                for i in range(len(self.childs)):
                    onnx_model = onnxmltools.convert_lightgbm(self.childs[i].model, initial_types= self.initial_types,target_opset=15,zipmap=False)
                    export_path = join(os.path.dirname(self.childs_path[i]),'onnx_model.onnx')
                    with open (export_path,'wb') as f:
                        f.write(onnx_model.SerializeToString())
            #TODO: For NeuralTorch model, there is a sklearn ColumnTransformer 
            # data processing in the predict method, and one of it's pipeline contains a QuantileTransformer which not available to convert to onnx. 
            # elif isinstance(self.childs[0].model,EmbedNet):
                # self.child_type = EmbedNet
                # model_input_sample = {
                #    'data_batch': {'vector': torch.zeros(size=(batch_size, feature_dim), dtype=torch_type)}}
                # for i in range(len(self.childs)):
                #     export_path = join(os.path.dirname(self.childs_path[i]),'onnx_model.onnx')
                #     torch.onnx.export(self.childs[i],
                #     args=model_input_sample,
                #     f='torch_nn.onnx',
                #     opset_version=15,
                #     do_constant_folding=True,
                #     )

    def rename_node_name(self):
        """
        This function is used to rename the node name of onnx models.
        """
        for i in range(len(self.onnx_models)):
            graph = self.onnx_models[i].graph
            for initializer in graph.initializer:
                initializer.name = initializer.name + str(i)
            for z in range(len(graph.output)):
                graph.output[z].name = graph.output[z].name + str(i)
            nodes = graph.node
            for j in range(len(nodes)):
                nodes[j].name = nodes[j].name + str(i)
                if j == 0:
                    for k in range(len(nodes[j].output)):
                        nodes[j].output[k] = nodes[j].output[k] + str(i)
                else:
                    for k in range(len(nodes[j].input)):
                        nodes[j].input[k] = nodes[j].input[k] + str(i)
                    for k in range(len(nodes[j].output)):
                        nodes[j].output[k] = nodes[j].output[k] + str(i)
                    
    def merge_onnx_models(self):
        """
        This function is used to merge the onnx models.
        """
        if self.childs:
            if isinstance(self.childs[0].model,EmbedNet):
                for onnx_model in self.childs:
                    graph = onnx_model.graph
                    graph.node.extend(self.operators['softmax'])
            self.merged_onnx_model = self.onnx_models[0]
            self.graph = self.merged_onnx_model.graph
            for i in range(1,len(self.onnx_models)):    
                for node in self.onnx_models[i].graph.node:
                    self.graph.node.extend([node])
                for initializer in self.onnx_models[i].graph.initializer:
                    self.graph.initializer.extend([initializer])
        else:
            raise Exception("No onnx models found.")
        
    def making_nodes(self):
        """
        This function is used to make nodes for the merged onnx model.
        """

        if self.child_type == lgbmBooster:
            self.operators = {'mean': mean_operator('result', 'probabilities',len(self.childs)), 
                              'argmax': argmax_operator('final_output', 'result')}
            # mean = onnx.helper.make_node(
            #         "Mean",
            #         inputs=["probabilities"+ str(i) for i in range(len(self.onnx_models))],
            #         outputs=["result"])
            # argmax = onnx.helper.make_node(
            #         "ArgMax", inputs=["result"], outputs=["final_output"], axis=1, keepdims=0)
            # self.graph.node.extend([mean,argmax])

        elif isinstance(self.childs[0].model,EmbedNet):
            self.operators = {'softmax': softmax_operator('after_softmax', 'output'), 
                              'mean': mean_operator('result', 'after_softmax',len(self.childs)), 
                              'argmax': argmax_operator('final_output', 'result')}
            # softmax = onnx.helper.make_node(
            #         "Softmax",
            #         inputs=["20"],
            #         outputs=["after_softmax"])
            
            # mean = onnx.helper.make_node(
            #     "Mean",
            #     inputs=["after_softmax0","after_softmax1","after_softmax2","after_softmax3","after_softmax4"],
            #     outputs=["result"])
            
            # argmax = onnx.helper.make_node(
            #     "ArgMax", inputs=["result"], outputs=["final_output"], axis=1, keepdims=0)
            
    def modify_graph(self):
        """
        This function is used to modify bagged model output and adding nodes.
        """

        if self.child_type == lgbmBooster:
            self.graph.node.extend([self.operators['mean'],self.operators['argmax']])
            self.graph.output[1].name = "final_output"
            del self.graph.output[0]
            self.merged_onnx_model.opset_import[0].version = 17
            self.graph.output[0].type.tensor_type.elem_type = 7
        elif self.child_type == EmbedNet:
            self.graph.nodes.extend(self.operators['mean'],self.operators['argmax'])
            self.graph.output[0].name = "final_output"
            self.graph.output[0].type.tensor_type.elem_type = 7

    def transform(self):
        """
        transfer and save the bagged model to onnx format.
        """
        self.save_child_onnx()
        self.making_nodes()
        self.child_to_onnx()
        self.rename_node_name()
        self.merge_onnx_models()
        self.modify_graph()
        with open (join(os.path.dirname(self.bagged_model_path),'onnx_model.onnx'),'wb') as f:
            f.write(self.merged_onnx_model.SerializeToString())

    def test(self,test_data):
        """
        This function is used to test the onnx model.
        """
        if self.child_type == lgbmBooster:
            sess = ort.InferenceSession(self.merged_onnx_model.SerializeToString())
            test = {}
            test = {'input0':test_data.astype(np.float32)}
            # for i in range(len(self.input_format)):
            #     print(len(self.input_format[i][1]))
                # test['input'+str(i)] = np.random.rand(1,len(self.input_format[i][1])).astype(np.float32)
            res = sess.run(None,test)
            return res

if __name__ == "__main__":
    args = get_args()
    bagged_model = bagged_model2onnx(args.model_dir)
    bagged_model.transform()
