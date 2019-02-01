# Generating adversrial image with Tensorflow #

This project is about training neural networks with Mnist data and generating corresponding adversarial image attack.

In *mnist_deep.py*, a neural network which classifies Mnist images with an accuracy higher than 99% is produced and saved in *nets*. 

In *mnist_deep_adversarial.py* the network is reconstructed and reloaded. Then I applied the **fast gradient sign targeted attack** to the network, and generated an image that gets classified into something different.

Other files:<br />
*batch_play.py*: exploring how Mnist dataset is formatted, and how to use batches. <br />
*mnist_softmax.py*: a simpler training with Mnist data. The resulting network has an accuracy of around 92%. <br />
*restore_sample_code.py*: exploring how Saver object in Tensorflow works, and experimenting on how to save and restore a net.

This project is a part of my research with the Security and Privacy Lab in Paul G. Allen school of CSE, UW. My log for this research with can be found at: <br />
https://medium.com/@yuji_cs/machine-learning-security-research-log-ad57434e1872 <br />
Check it out! <3
