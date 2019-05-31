# Note (by Sasaki)

This is another implementation of VQA based on https://github.com/Shivanshu-Gupta/Visual-Question-Answering.

To execute the code, try these at the experiment2 directory:

```
pip install --user -r requirements.txt
mkdir results
mkdir preprocessed
mkdir preprocessed/img_vgg16feature_train
mkdir preprocessed/img_vgg16feature_val
python main.py -c config.yml
```

To skip preprocessing after the first execution, disable 'preprocess' in the config file.

The followings are the original README:

# Visual-Question-Answering
This repository contains an AI system for the task of **[Visual Question Answering]**: given an image and a question related to the image in natural language, the systems answer the question in natural language from the image scene. The system can be configured to use one of 3 different underlying models:

1. **VQA**: This is the *baseline model* given in the paper [VQA: Visual Question Answering]. It encodes the image by a CNN and the question by an LSTM and then combines these for VQA task. It uses *pretrained vgg16* to get the image embedding (may be further normalised), and a 1 or 2-layered LSTM for the question embedding.
2. **SAN**: This is an *attention based model* described in the paper [Stacked Attention Networks for Image Question Answering]. It incorporates attention on the input image.
3. **MUTAN**: This is a variant of the VQA model where instead of a simple of pointwise-product, the image and question embedding are combined using a a special *Multimodal Tucker fusion* technique described in the paper [MUTAN: Multimodal Tucker Fusion for Visual Question Answering].

## Usage
First download the datasets from [http://visualqa.org/download.html] - all items under *Balanced Real Images* except *Complementary Pairs List*. 
```sh
python main.py --config <config_file_path>
```
The system takes its arguments from the config file that it takes as input. Sample config files have been provided in [config/].

In order to speed up the training, it's possible to preprocess the images in the dataset and store the image embeddings by setting the *emb_dir* and *preprocess* flag.

[Visual Question Answering]: https://vqa.cloudcv.org/
[VQA: Visual Question Answering]: https://arxiv.org/abs/1505.00468
[Stacked Attention Networks for Image Question Answering]: https://arxiv.org/pdf/1511.02274
[MUTAN: Multimodal Tucker Fusion for Visual Question Answering]: https://arxiv.org/abs/1705.06676
[http://visualqa.org/download.html]: http://visualqa.org/download.html
[config/]: https://github.com/Shivanshu-Gupta/Visual-Question-Answering/config

We require the following for the below error:

pip install --user -U protobuf

abati@abati-23031:~/ECE285/experiment2$ cat RESULTTXT
nohup: ignoring input
Traceback (most recent call last):
  File "main.py", line 12, in <module>
    from train import train_model
  File "/datasets/home/home-02/59/659/abati/ECE285/experiment2/train.py", line 3, in <module>
    from tensorboardX import SummaryWriter #sasaki
  File "/datasets/home/59/659/abati/.local/lib/python2.7/site-packages/tensorboardX/__init__.py", line 5, in <module>
    from .torchvis import TorchVis
  File "/datasets/home/59/659/abati/.local/lib/python2.7/site-packages/tensorboardX/torchvis.py", line 11, in <module>
    from .writer import SummaryWriter
  File "/datasets/home/59/659/abati/.local/lib/python2.7/site-packages/tensorboardX/writer.py", line 27, in <module>
    from .event_file_writer import EventFileWriter
  File "/datasets/home/59/659/abati/.local/lib/python2.7/site-packages/tensorboardX/event_file_writer.py", line 28, in <module>
    from .proto import event_pb2
  File "/datasets/home/59/659/abati/.local/lib/python2.7/site-packages/tensorboardX/proto/event_pb2.py", line 15, in <module>
    from tensorboardX.proto import summary_pb2 as tensorboardX_dot_proto_dot_summary__pb2
  File "/datasets/home/59/659/abati/.local/lib/python2.7/site-packages/tensorboardX/proto/summary_pb2.py", line 15, in <module>
    from tensorboardX.proto import tensor_pb2 as tensorboardX_dot_proto_dot_tensor__pb2
  File "/datasets/home/59/659/abati/.local/lib/python2.7/site-packages/tensorboardX/proto/tensor_pb2.py", line 15, in <module>
    from tensorboardX.proto import resource_handle_pb2 as tensorboardX_dot_proto_dot_resource__handle__pb2
  File "/datasets/home/59/659/abati/.local/lib/python2.7/site-packages/tensorboardX/proto/resource_handle_pb2.py", line 22, in <module>
    serialized_pb=_b('\n(tensorboardX/proto/resource_handle.proto\x12\x0ctensorboardX\"r\n\x13ResourceHandleProto\x12\x0e\n\x06\x64\x65vice\x18\x01 \x01(\t\x12\x11\n\tcontainer\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x11\n\thash_code\x18\x04 \x01(\x04\x12\x17\n\x0fmaybe_type_name\x18\x05 \x01(\tB/\n\x18org.tensorflow.frameworkB\x0eResourceHandleP\x01\xf8\x01\x01\x62\x06proto3')
TypeError: __new__() got an unexpected keyword argument 'serialized_options'
