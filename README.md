## Setup
1. Install required packages by running
    ```
    pip3 install -r requirements.txt
    ```

## TranSalNet.py
1. You can pass in two directories(generated_image and mask), and assign a output directory, it will generate the predicted outcome in .json format to the output directory.
    ```
    python TranSalNet.py --image "image_folder" --mask "mask_folder" --output "output_folder"
    ```

## product_hole_issue.py
1. You can change the **image_path** and **ori_mask_path** in the file and get the one shot result
