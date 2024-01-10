## Setup
1. Install required packages by running
    ```
    pip3 install -r requirements.txt
    ```
    
## hole_issue_detection.py
1. You can pass in two directories(generated_image and mask), and assign a output directory, it will generate the predicted outcome in .json format to the output directory.
    ```
    python hole_issue_detection.py --image "image_folder" --mask "mask_folder" --output "output_folder"
    ```
    
2. Otherwise, you can simply use the "hole_issue_detection" function. Pass in a image_path and a mask_path, it will generate 0 or 1 where 0 means without hole_issue, 1 means with hole_issue.
    ```
    hole_issue_detection(image_path, mask_path) -> 1 or 0
    ```

