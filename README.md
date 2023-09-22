# Labeling Tool
Created by Felix Liawi to support image classification labeling.

## Setup
1. Install required packages by running
    ```
    pip3 install -r requirements.txt
    ```

2. Move the downloaded data to `static` directory, for example: `static/generated_image`, `static/mask`, `static/real_image`, `static/prompt.csv`

3. Run `divide_dataset.py` to divide the dataset, to make sure that you only label the your portion. Before running the python file, please fill your id in line 7.
    ```
    python3 divide_dataset.py
    ```

4. Run `app.py` to start labeling by running
    ```
    python3 app.py
    ```
