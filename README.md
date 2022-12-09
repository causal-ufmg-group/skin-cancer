# **skin-cancer**

## **Getting Started**
---

Python version: 3.10

### **Conda**

If you use the conda package manager, you can create a virtual environment with the provided environment.yml file:

```
conda env create -f environment.yml
```

Then, you can activate it by running:
```
conda activate skin_cancer
```

## **Motivation**
---

[Domain Generalization using Causal Matching](https://www.microsoft.com/en-us/research/uploads/prod/2021/06/DG_with_causal_matching.pdf) is a really interesting paper regarding domain generalization using causality. 

However, trying to extend their implementation to other datasets is not an easy task because their code - found in [RobustDG](https://github.com/microsoft/robustdg) - is not particularly clear about some detail, namely, (a) there isn't any detail about what kinda of arguments their methods expect; and (b) the paper goes over a bunch of different dataset and neural network architectures so it felt kinda hard to filter out exactly what we need.

Regardless, after trying quite a bit this repository *should* overall serve the same purpose as (part) of their implementation. A lot of the code is still from their work, we just made some changes (in [Changes](#changes)) and documented some functions. 

While their original github page is linked as a submodule, it really isn't necessary to run any of our (slightly) modified version. It is there mostly for reference.

A lot could still be done to improve it.

### **Reference**

A lot of the configuration were taken from:

- `robustdg/reproduce_scripts/mnist_run.py`
- `robustdg/reproduce_scripts/pacs_run.py`
- `robustdg/docs/reproduce_results.ipynb`

Our dataset classes (described in [Changes](#changes)) were based off of:

- `robustdg/data/data_loader.py`
- `robustdg/data/mnist_loader.py`
- `robustdg/data/pacs_loader.py`
- `robustdg/utils/helper.py (get_dataloader function)`

Other files were mostly based off robustdg's files with the same name.

## **How to use**
---

### **Configuration**

General configurations can be found in the `src/robustdg_modified/config/` directory. 

- Some hyper parameters are defined there.
    - `hyper_parameters.py`: Really general parameters. Most are even overwritten by algorithm specific parameters
    - `algorithms.py`: Hyper parameters used in the original paper for some cases.
- [dataset](#dataset) specific paths mentioned from here on out are defined in `paths.py`.
- General RobustDG arguments (`args_mock.py`) are also there (see [Changes](#changes)).
- Reproducibility (function to set seeds correctly) are in `reproducibility.py`

### **Notebooks**

One of the main reasons, we modified (see [Changes](#changes)) their work was to allow the use of jupyter notebooks. They can be divided into two main categories:

**Specific:**

- `src/data_aumentation.ipynb` and `src/data_augmentation/`
    - Useful for unbalanced datasets
    - It was used for some simple data augmentation to balance our skin-cancer train dataset a bit. 
        - The main goal is to make all (domain, labels) pairs have comparable number of images.
    - See [Dataset](dataset)

- `src/test_dataset.ipyny`

    - Specific for our skin-cancer dataset since our test and train dataset intersected with each other.
    - Used to remove intersection and reduce the size of our test dataset.
    - See [Test Dataset](#test-dataset)

**General use:**
- `src/train.ipynb`
    - Procedure for training differente model.
    - There are *templates* for all algorithms there.
        - Configurations can be found in `src/robustdg_modified/config/algorithms/`
- `src/test.ipynb`
    - Some simple test for a neural network
        - Checkpoint name will have to be provided
- `src/dataset.ipynb`
    - Describes some general information about the dataset.

### **Logging**

Logging information will be stored in `logs/`.

- Logging information (see [Changes](#changes)) will be stored in `logs/all_logs.log`.
- Pytorch neural network save states will be stored inside
`logs/checkpoints`.

### **Dataset**

Dataset should be stored inside `data/` directory.

Specifically, the following is expected:

- `data/csv_files/`
    - Should contain all following .csv files from ISIC.
        - Ground Truth from ISIC2017 (Task 2), ISIC2018 (Task 3), ISIC2019 (Task 1)
        - Lesion Groupings from ISIC2018 (Task 3): (Supplemental Information)

- `data/ISIC2018_Task3_Training_Input/`

    - Should contain images from ISIC2018 task 3 dataset.
    - This is the default folder name after downloading.

- `data/ISIC2019_Training_Input/`

    - Should contain images from ISIC2019
    - This is the default folder name after downloading.

Datasets used can be found in the following webpages:
- [ISIC2017 - Task 2](https://challenge.isic-archive.com/data/#2017)
- [ISIC2018 - Task 3](https://challenge.isic-archive.com/data/#2018)
- [ISIC2018 - Supplemental Information](https://forum.isic-archive.com/t/task-3-supplemental-information/430)
- [ISIC2019 - Task 1](https://challenge.isic-archive.com/data/#2019)


### **Test Dataset**

We are using ISIC2019 image as test dataset, but as some images are from ISIC2018 and ISIC2017 we filter them out in `src/test_dataset.ipynb`

## **Changes**
---

- Created a mock for their argument parses, so that it can be easily run in a notebook rather than through shell commands. 
    - See src/robustdg_modified/config/args_mock.py for some reference.
    - You can see src/train.ipynb for the entire training procedure.

- Print statements were changed by logging statements.

- Created some dataset classes which should be clear in what they expect as parameters. We also provide constructor functions to create them which automatically calculate some parameters required for their algorithms.
    - However, it is assumed that the object is known (similar to MNIST in their paper). Some modification could be done to make it general, like adding a flag, but it has done been done yet.

- Dataset index information is also returned when indexing. 
    - This is mostly done to avoid loading all images into memory was as done originally. 
    - Only dataset indexes are stored now. So, image should only be loaded when needed.

- Their algorithms were changed slightly to allow for empty values in the class/domain table used. While comparing values from different domains, only if both exist their distance will be calculated.
    - This requires more testing.

- Only test after ideal model has been decided. 
    - While the original work did not use test information to decide the best model (as it should be), they would still compute it for each epoch which is computationally wasteful.
