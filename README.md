# scene-annotator
Annotate an RGB-D scene with background/foreground labels using ground truth 3D poses.

Make sure to install CoViS from [here](https://gitlab.com/caro-sdu/covis) and don't forget to compile the Python bindings using **make covis_python**.

You need the following Python dependencies for this script:
- numpy
- scikit-image

And finally you need to make sure that CoViS is built "one folder up" relative to this directory - otherwise, you have to change the line below the import statement near the top of [scene_annotator.py](scene_annotator.py) to append the correct path to the bindings.

## Testing
The script is tailored to load test sequences from the SIXD Challenge datasets. If you download one of the datasets from the [webpage](http://cmp.felk.cvut.cz/sixd/challenge_2017) (you only need the **models.zip** and **test.zip** files from each dataset), you need to tell the script where to find the root of the dataset, what object to annotate, what scene sequence to process, and finally where to output the training data:
```sh
python scene_annotator.py /path/to/sixd_dataset/ --object=models/obj_01.ply --scene_dir=test/01 --output-dir=./output
```

