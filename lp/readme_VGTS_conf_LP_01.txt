
1) The python scrpit LP is hard-coded to the below model. It is done for the below


----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1             [-1, 32, 6, 6]           4,032
              ReLU-2             [-1, 32, 6, 6]               0
            Conv2d-3             [-1, 64, 6, 6]          51,264
              ReLU-4             [-1, 64, 6, 6]               0
            Linear-5                  [-1, 256]         590,080
              ReLU-6                  [-1, 256]               0
            Linear-7                  [-1, 128]          32,896
              ReLU-8                  [-1, 128]               0
            Linear-9                  [-1, 128]          16,512
             ReLU-10                  [-1, 128]               0
           Linear-11                    [-1, 1]             129
             ReLU-12                    [-1, 1]               0
           Linear-13                  [-1, 128]          16,512
             ReLU-14                  [-1, 128]               0
           Linear-15                    [-1, 6]             774
======================================================

3) Install CPLEX. I used version 12.10.  Students can get the full version free: https://content-eu-7.content-cms.com/b73a5759-c6a6-4033-ab6b-d9d4f9a6d65b/dxsites/151914d1-03d2-48fe-97d9-d21166848e65/academic/home


4) Setup CPLEX's Python
   See https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html
Similar to below commands are used:

       cd ~/CPLEX_Studio1210/cplex/python/3.7/x86-64_linux
       python setup.py install

5) shape list containts the shapes (sizes) of each tensor in the CNN (line 106) - change accordingly 


6) From torch CNN, replace below and put it inside a numpy array
- weights and biases and strides of conv1.  line 203
- weights and biases and strides of conv2.  line 270
- weights and biases of dense1. line 586
- weights and biases of dense2. line 633
- weights and biases of dense3. line 681
- weights and biases of dense4. line 729
- weights and biases of dense5. lines 812 and 861 

7) Testing the output of the LP (the confusing image) - printing the output probabilities from the torch CNN trained model. In other words, input the output tensor of the LP to the torch model and print the logit (logit is the list of probabilities).

