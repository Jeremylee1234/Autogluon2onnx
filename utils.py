import os
from os.path import isfile, isdir, join

def model_dir_tools(model_dir):
    """
    This function is used to get the model path and child model path.
    Parameters
    ----------
    model_dir : str
        Path to the bagged_model in autogulon directory.
    Returns
    -------
    bagged_model_path : str
        Path to the bagged_model.pkl.
    childs_dir : list
        List of child model path.
    """
    
    bagged_model_path = join(model_dir,'model.pkl')
    childs_dir = os.listdir(model_dir)
    childs_new_dir = []
    for child_dir in childs_dir:
        if child_dir != 'utils':
            if isdir(join(model_dir,child_dir)):
                child_dir = join(model_dir,child_dir,'model.pkl')
                childs_new_dir.append(child_dir)
    return bagged_model_path,childs_new_dir