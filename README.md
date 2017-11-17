### VietnameseOCR - Vietnamese Optical Character Recognition

Apply Deep Learning ( CNN networks ) to train a model uses for recognizing Vietnamese characters, it works well with Latin characters.

### Dataset in big image ( 100.000 samples, 2800 x 2800 pixel)


![](data/vocr_dataset.png)


### Requirements
```
python 3.6.5
tensorflow
PIL
```


### Model Summary

| Layer         | Shape 		 | 	Kernel   	  |    Stride 	  |   Padding 	|   	 	 |
| -------------:| --------------:|---------------:|--------------:|------------:|-----------:|
| INPUT     	| [28, 28, 1] 	 |			   	  | 			  |				|			 |
| CONV1			| 				 | [3, 3, 32, 32] |  	[1, 1]    |    SAME     |   	 	 |
| POOL1         |				 |				  |               |				|			 |
| CONV2		    |				 | [3, 3, 32, 64] |     [1, 1]    |	   SAME		|			 |
| POOL2			|				 |                |               |				|			 |
| CONV3			| 				 | [3, 3, 64, 128]|     [1, 1]    |	   SAME		|			 |
| POOL3			|				 |                |               |				|			 |
| FC1			| 				 |                |               |				|			 |
| FC2			| [625, 190]	 |                |               |				|			 |


### Results
```
Training...
Epoch: 01 cost = 6.772273257
Epoch: 02 cost = 4.885908848
Epoch: 03 cost = 4.292183308
Epoch: 04 cost = 3.791747945
Epoch: 05 cost = 3.244293085
Epoch: 06 cost = 2.757302330
Epoch: 07 cost = 2.223773586
Epoch: 08 cost = 1.844299474
Epoch: 09 cost = 1.571607014
Epoch: 10 cost = 1.377784024
Epoch: 11 cost = 1.216272550
Epoch: 12 cost = 1.091827986
Epoch: 13 cost = 0.973539455
Epoch: 14 cost = 0.879507029
Epoch: 15 cost = 0.813665386

Accuracy Evaluates
------------------------------
Test Accuracy: 0.927347593306
```


### Training
#### Prepare dataset for training
```
git clone https://github.com/miendinh/VietnameseOCR.git
cd VietnameseOCR/data/train/characters
unzip dataset.zip
```

#### Let's train.
```
python train.py
```


#### Create you own dataset
##### Prepare fonts for generating text-image
- You can add more fonts
```
cd VietnameseOCR/data/train/characters
unzip google.zip
unzip win.zip
```
##### Create font list which save in file fonts.list
```
source ./list.sh
```

##### Generate Text Images Dataset
```
python generate_data.py
```

### Play with pretrained model
- All pretrained weights of model is save to file vocr.brain
- Let's test with random character in dataset
```
python predict.py
```

### Further working
- Character classification. -> Done.
- Dataset augmentation.     
- Improve accuracy.
- Text location.
- Text recognition.
- Apply NLP for spell checking.

### References

1. [STN-OCR: A single Neural Network for Text Detection and Text Recognition](https://arxiv.org/abs/1707.08831)
2. [Automatic Dataset Augmentation](https://arxiv.org/abs/1708.08201)
3. [VGG16 implementation in TensorFlow](http://www.cs.toronto.edu/~frossard/post/vgg16/)

### Author mien.hust [at] gmail.com