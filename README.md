# circle-rectangle-classifier

## DataSet

The dataset is has colored 3 channel (RGB) pictures and I don't think it serves any purpose in classifying them into circles and rectangles. So the first thing I am doing with the pictures is turn them to a single channel grayscaled image. So now the shape will be (64,64) which was (64,64,3) before. After this the images are also normalized (x = x/255)

## NeuralNetwork

My approach is to keep the network as simple as possible to match up with the simplicity of the problem. The neural network with start off with a Conv2D with 32 filters which is a followed by a MaxPool2D. After this the layers are flattened into one dimension and followed by a fully connected dense layer of 16 units which will output to a sigmoid unit.

I will be using Binary Categorical Crossentropy as the loss function because this is a binary classification problem.