###  dataset structure
```
-deepscores
    -deep_scores_v2_100p
        -images_png
            -~~~~.png
            -~~~~.png
        -meta_info
            -~~~~.png
            -~~~~.png
        -pix_annotations_png
            -~~~~.png
            -~~~~.png
        -xml_annotations
            -~~~~.png
            -~~~~.png
    -test_output
-train_files.txt
-valid_files.txt
-test_files.txt
```

### train

first. run split_train_valid_test.ipynb to make txt files
second. run train.py
