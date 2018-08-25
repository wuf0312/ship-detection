# ship-detection

The code of our method.


1. Requirements: software
   1) Requirements for Caffe and pycaffe 
      (about how to install Caffe and pycaffe, please see http://caffe.berkeleyvision.org/installation.html for detail)
      
   2) Recommend to install Anaconda
   
   
2. Requirements: hardware
   1) a GPU (e.g., 1080, 1080Ti, Titan, K20, K40, ...) with at least 3G of memory
      (we use a 1080Ti, and a GPU with at least 3G of memory should work well although we have not tested it)
      
      
3. Demo 
   1) get the model for detection: we have provided the models for ship head classification and ship localization, please
   
      ~$: cd code
      
      ~$: ipython notebook
      
      then open merge_model.ipynb and run, you will get the model for ship detection
      
   2) To run the demo:
   
      ~$: cd code
      
      ~$: python tools/demo.py
      
                                     
