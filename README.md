# OSCAR-Net:  Object-centric Scene Graph Attention for Image Attribution
This repo contains the demo code to run our OSCAR-Net model.  

We also provide the dataset [IDs, 50mb](https://cvssp.org/data/Flickr25K/tubui/cvpr21wmf/stock4_7M.txt) for the 4.7 million stock images from Adobe.

See Adobe [APIs](https://www.adobe.io/apis/creativecloud/stock/docs.html) on how to retrieve the images.


<!-- Please find the link to code at [link\_to\_code.txt](link\_to\_code.txt) which also stores a pretrained model. -->


## Requirements
<!-- Docker version >= 19.03 -->
<!-- Nvidia-docker2 >= 2.0.3 -->
Nvidia driver >= 418.39  
setuptools >= 41.0.0  
h5py >= 2.9.0  
h5py-cache >= 1.0  
opencv-python >= 4.2.0  
pandas >= 0.24.1  
scikit-image >= 0.15.0  
tqdm >= 4.43.0  
reportlab >= 3.5.23  
numpy >= 1.16.4  
scipy >= 1.4.1  
requests >= 2.22.0  
cython  

A GPU with compute capability >= 3.0 and at least 8GB GPU memory.  


### Download

Download the `weight` zip from [here](https://drive.google.com/file/d/1kGK-s5M5mLEYuFkGv6GKrp3bB8tWv27s/view?usp=sharing), and put the contents into the project `weight` directory (i.e., replace the `weight` directory).


### Optional: dockerfile provided


### Run inference on an image:

```
python inference.py -i examples/original.jpg -w weight/best.pt
```

This should output a 64-bit hash code.

### Run the demo

```
python demo.py
```

This demo loads an original image [docs/examples/original.jpg](docs/examples/original.jpg), a benign-transformed version [docs/examples/benign.jpg](docs/examples/benign.jpg) and a manipulated version [docs/examples/manipulated.jpg](docs/examples/manipulated.jpg) of that image; then compare the Hamming distance of the original-benign and original-manipulated pairs.

The output should look like this:
```
Hamming (original.jpg, benign.jpg): 3
Hamming (original.jpg, manipulated.jpg): 22
```

Original | Benign transform | Manipulated
:---: | :---: | :---:
<kbd><img src="docs/examples/original.jpg" height="300px"/></kbd> | <img src="docs/examples/benign.jpg" height="300px"/> | <img src="docs/examples/manipulated.jpg" height="300px"/>