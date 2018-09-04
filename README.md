# ship-detection

The code of our method.


1. Requirements: software
   1) Requirements for Caffe and pycaffe
   
      (about how to install Caffe and pycaffe, please see http://caffe.berkeleyvision.org/installation.html for detail)
      
   2) Recommend to install Anaconda
   
   
2. Requirements: hardware
   1) a GPU (e.g., 1080, 1080Ti, Titan, K20, K40, ...) with at least 3G of memory
   
      (we use a 1080, and a GPU with at least 3G of memory should work well although we have not tested it)
      
      
3. Demo 
   1)   ~$: cd code/lib
   
        ~$: make
   
   2) Get the model for detection. We have provided the models for ship head classification and ship localization, please
   
        ~$: cd code
      
        ~$: ipython notebook
      
      then open merge_model.ipynb and run, you will get the model for ship detection (you will get two models: demo.caffemodel, and demo_cascade_2.caffemodel.)
            
       (Notice: The two models for ship detection are both large than 100M, and in our previous version of code, it seems that the models were not uploaded correctly since such files exceed GitHub's file size limit of 100.00 MB. In this verison of code, you can do as shown as above to get the models for detection.)
      
           
   3) Then run the demo:
       
        ~$: python tools/demo.py
      
                                     
