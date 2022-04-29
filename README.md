# CommNet+: EECS 692 Team 3 2022
This is a significantly modified version of the IC3Net code repository referenced at the bottom. Added functionality and easy level editing.

## Installation
This code repository is designed to function on python 2.7.

First, clone the repo and install ic3net-envs which contains implementation for Predator-Prey and Traffic-Junction

```
git clone https://github.com/grantzl/CommNet_EECS_692.git
cd CommNet_EECS_692/ic3net-envs
python setup.py develop
```

Downloading torch version 0.4.0
https://pytorch.org/get-started/previous-versions/
    Down close to the bottom there is the via pip section, there are a bunch of html links. 
    I used CPU-only build, but I would assume the CUDA version would work too
    https://download.pytorch.org/whl/cpu/torch_stable.html
    
    Find the torch version you need
    Make sure its 0.4.0, cp27mu, and linux
    Here is the one I used: https://download.pytorch.org/whl/cpu/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl
    
    pip install filename.whl

Install the following files IN ORDER. It would overwrite other packages to newer versions if I didn't do it in this order
    pip install Pillow==6.2.0
    pip install numpy==1.13.3
    pip install visdom==0.1.4
    pip install gym
    
    Afterwards you can pip install whatever errors pop up. Should only be one or two packages
    
    DOUBLE CHECK THAT YOU HAVE THE CORRECT VERISONS INSTALLED OR IT WILL MOST LIKELY SYNTAX ERROR YOU
        pip list

## Running

Once everything is installed, we can run the tasks. All of the tasks for Predator Prey variants along with their specifications are contained within the Grids folder. You can create new tasks by creating a new Info_numrowxnumcol_Name folder with a Grid_numrowxnumcol_Name.csv file inside of it. You want to be inside of the CommNet_EECS_692 folder when running these commands


### Predator-Prey
Feel free to modify the parameters as needed, but here is a sample 5x5_PP. Modified parameters are detailed in the specific task info folders
  python main.py --env_name predator_prey --nagents --nprocesses 1 --num_epochs 10 --hid_size 128 --detach_gap 10 --lrate 0.001 --max_steps 20 --commnet --vision 2 --recurrent --load_grid 5x5_PP --save 5x5_PP

If you want to use the visdom plots run the following command in a second terminal first
  python -m visdom.server —port 8097

then add the following arguments to the main run: --plot --plot_env main

You can use the --display argument to visualize one training batch per epoch

### Traffic Junction
Feel free to modify the parameters as needed, the ones within scalability.ipynb are the recommended ones from the CommNet repository. Plotting can be seen by using the --plot flag, and going to the URL printed out in the "Setting Up Visdom" section. The last three cells in the notebook handle linearly increasing the number of junctions & agents using a bash for-loop. The plot_scalability notebook takes in .txt files in the formal "#;Success" to plot the success rate over the number of agents/junctions.


# IC3Net
This repository contains reference implementation for IC3Net paper (accepted to ICLR 2019), **Learning when to communicate at scale in multiagent cooperative and competitive tasks**, available at [https://arxiv.org/abs/1812.09755](https://arxiv.org/abs/1812.09755)

Github: https://github.com/ic3net/ic3net#start-of-content

```
@article{singh2018learning,
  title={Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks},
  author={Singh, Amanpreet and Jain, Tushar and Sukhbaatar, Sainbayar},
  journal={arXiv preprint arXiv:1812.09755},
  year={2018}
}
```
