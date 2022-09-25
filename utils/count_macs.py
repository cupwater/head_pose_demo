import onnx_tool
modelpath = 'weights/lite_hrnet_30_coco.onnx'
onnx_tool.model_profile(modelpath, None, None) # pass file name
onnx_tool.model_profile(modelpath, savenode='node_table.txt') # save profile table to txt file
onnx_tool.model_profile(modelpath, savenode='node_table.csv') # save profile table to csv file