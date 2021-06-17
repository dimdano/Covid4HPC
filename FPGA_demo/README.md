## FPGA demo

First, make sure you followed the setup instructions and activate conda vitis-ai tensorflow. Then, to run demo execute:  

```bash
/usr/bin/python3 run_inference.py -m model_dir/CustomCNN.xmodel -t 6 
  ```
The FPS achieved will be shown and a ".txt" file will be generated with the results.  
