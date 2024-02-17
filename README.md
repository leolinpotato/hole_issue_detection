## Setup
1. Install required packages by running
    ```
    pip3 install -r requirements.txt
    ```

2. Download pretrained_models from https://github.com/LJOVO/TranSalNet and place it into 
    ```
    code/TranSalNet/pretrained_models
    ```
## TranSalNet.py
1. You can pass in two directories(generated_image and mask), and assign a output directory, it will generate the predicted outcome in .json format to the output directory.
    ```
    python TranSalNet.py --image "image_folder" --mask "mask_folder" --output "output_folder"
    ```