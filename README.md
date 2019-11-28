## Homework 3 - Deep Q-Learning
This repo contains implementation of a deep Q-learning network to train an agent for the [MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/). The details about the network and hyperparameters can be found in the [submission report](HW3_CMSC818B.pdf).

## Dependencies
To run the code you require the following dependencies:
- python 3.5+
- `OpenAI gym` : To install use
```
sudo -H pip3 install gym
``` 
If you face any troubles installing follow instructions [here](http://gym.openai.com/docs/)

- `PyTorch` : To install pytorch for python 3.5+ use
```
sudo -H pip3 install torch torchvision
```
For any issues read official documentation [here](https://pytorch.org/get-started/locally/)

- `TensorboardX`: If you want to view pytorch plots in tensorboard install [tensorboardX](https://pypi.org/project/tensorboardX/)
```
sudo -H pip3 install tensorboardX
```
## Running the code 
To run the code download the repository and keep the directory structure intact. There is also a pre-trained model, which was trained for about 15K iterations. Using the following command will automaticaly load the model and show the inference run for the model:
```
python3 problem1_sol.py
```
To start the training from beginning for a new model use
```
python3 problem1_sol.py --train=True --load_model=False
```
## Results
The checkpoint files for trained models are available in the `models` subdirectory. A [video](mountainCar.mp4) showing progression of the training from random policy to final optimal policy is also included.
![Training](agent.gif)