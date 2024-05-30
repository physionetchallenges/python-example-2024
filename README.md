# Python example code for the George B. Moody PhysioNet Challenge 2024

## What's in this repository?

This repository contains a simple example that illustrates how to format a Python entry for the [George B. Moody PhysioNet Challenge 2024](https://physionetchallenges.org/2024/). If you are participating in the 2024 Challenge, then we recommend using this repository as a template for your entry. You can remove some of the code, reuse other code, and add new code to create your entry. You do not need to use the models, features, and/or libraries in this example for your entry. We encourage a diversity of approaches for the Challenges.

For this example, we implemented a random forest model with several simple features. (This simple example is **not** designed to perform well, so you should **not** use it as a baseline for your approach's performance.) You can try it by running the following commands on the Challenge training set. If you are using a relatively recent personal computer, then you should be able to run these commands from start to finish on a small subset (1000 records) of the training data in less than 30 minutes.

## How do I run these scripts?

First, you can download and create data for these scripts by following the instructions in the following section.

Second, you can install the dependencies for these scripts by creating a Docker image (see below) or [virtual environment](https://docs.python.org/3/library/venv.html) and running

    pip install -r requirements.txt

You can train your model(s) by running

    python train_model.py -d training_data -m model

where

- `training_data` (input; required) is a folder with the training data files, including the images and classes (you can use the `ptb-xl/records500/00000` folder from the below steps); and
- `model` (output; required) is a folder for saving your model(s).

We are asking teams to include working training code and a pre-trained model. Please include your pre-trained model in the `model` folder so that we can load it with the below command.

You can run your trained model(s) by running

    python run_model.py -d test_data -m model -o test_outputs

where

- `test_data` (input; required) is a folder with the validation or test data files, excluding the images and classes (you can use the `ptb-xl/records500_hidden/00000` folder from the below steps, but it would be better to repeat these steps on a new subset of the data that you did not use to train your model);
- `model` (input; required) is a folder for loading your model(s); and
- `test_outputs` is a folder for saving your model outputs.

We are asking teams to include working training code and a pre-trained model. Please include your pre-trained model in the `model` folder so that we can load it with the above command.

The [Challenge website](https://physionetchallenges.org/2024/#data) provides a training database with a description of the contents and structure of the data files.

You can evaluate your model by pulling or downloading the [evaluation code](https://github.com/physionetchallenges/evaluation-2024) and running

    python evaluate_model.py -d labels -o test_outputs -s scores.csv

where

- `labels` is a folder with labels for the data, such as the training database on the PhysioNet webpage (you can use the `ptb-xl/records500/00000` folder from the below steps, but it would be better to repeat these steps on a new subset of the data that you did not use to train your model);
- `test_outputs` is a folder containing files with your model's outputs for the data; and
- `scores.csv` (optional) is file with a collection of scores for your model.

## How do I create data for these scripts?

You can use the scripts in this repository to generate synthetic ECG images for the [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6). You will need to generate or otherwise obtain ECG images before running the above steps.

1. Download (and unzip) the [PTB-XL dataset](https://physionet.org/content/ptb-xl/) and [PTB-XL+ dataset](https://physionet.org/content/ptb-xl-plus/). These instructions use `ptb-xl` as the folder name that contains the data for these commands (the full folder name for the PTB-XL dataset is currently `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`, and the full folder name for the PTB-XL dataset is currently `ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1`), but you can replace it with the absolute or relative path on your machine.

2. Add information from various spreadsheets from the PTB-XL dataset to the WFDB header files:

        python prepare_ptbxl_data.py \
            -i  ptb-xl/records500/00000 \
            -pd ptb-xl/ptbxl_database.csv \
            -pm ptb-xl/scp_statements.csv \
            -sd ptb-xl/12sl_statements.csv \
            -sm ptb-xl/12slv23ToSNOMED.csv \
            -o  ptb-xl/records500/00000

3. [Generate synthetic ECG images](https://github.com/alphanumericslab/ecg-image-kit/tree/main/codes/ecg-image-generator) on the dataset:

        python gen_ecg_images_from_data_batch.py \
            -i ptb-xl/records500/00000 \
            -o ptb-xl/records500/00000 \
            --print_header

4. Add the file locations and other information for the synthetic ECG images to the WFDB header files. (The expected image filenames for record `12345` are of the form `12345-0.png`, `12345-1.png`, etc., which should be in the same folder.) You can use the `ptb-xl/records500/00000` folder for the `train_model` step:

        python prepare_image_data.py \
            -i ptb-xl/records500/00000 \
            -o ptb-xl/records500/00000

5. Remove the waveforms, certain information about the waveforms, and the demographics and classes to create a version of the data for inference. You can use the `ptb-xl/records500_hidden/00000` folder for the `run_model` step, but it would be better to repeat the above steps on a new subset of the data that you will not use to train your model:

        python gen_ecg_images_from_data_batch.py \
            -i ptb-xl/records500/00000 \
            -o ptb-xl/records500_hidden/00000 \
            --print_header \
            --mask_unplotted_samples

        python prepare_image_data.py \
            -i ptb-xl/records500_hidden/00000 \
            -o ptb-xl/records500_hidden/00000

        python remove_hidden_data.py \
            -i ptb-xl/records500_hidden/00000 \
            -o ptb-xl/records500_hidden/00000 \
            --include_images

## Which scripts I can edit?

Please edit the following script to add your code:

* `team_code.py` is a script with functions for training and running your trained model(s).

Please do **not** edit the following scripts. We will use the unedited versions of these scripts when running your code:

* `train_model.py` is a script for training your model(s).
* `run_model.py` is a script for running your trained model(s).
* `helper_code.py` is a script with helper functions that we used for our code. You are welcome to use them in your code.

These scripts must remain in the root path of your repository, but you can put other scripts and other files elsewhere in your repository.

## How do I train, save, load, and run my models?

You can choose to create digitization and/or classification models.

To train and save your model(s), please edit the `train_models` function in the `team_code.py` script. Please do not edit the input or output arguments of this function.

To load and run your trained model(s), please edit the `load_models` and `run_models` functions in the `team_code.py` script. Please do not edit the input or output arguments of these functions.

## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies so that your code can be reliably run in other computational environments.

To increase the likelihood that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a small subset of the training data, such as 1000 records.

If you have trouble running your code, then please try the follow steps to run the example code.

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data test_data model test_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2024/#data). Put some of the training data in `training_data` and `test_data`. You can use some of the training data to check your code (and you should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

        user@computer:~/example$ git clone https://github.com/physionetchallenges/python-example-2024.git

4. Build a Docker image and run the example code in your terminal.

        user@computer:~/example$ ls
        model  python-example-2024  test_data  test_outputs  training_data

        user@computer:~/example$ cd python-example-2024/

        user@computer:~/example/python-example-2024$ docker build -t image .

        Sending build context to Docker daemon  [...]kB
        [...]
        Successfully tagged image:latest

        user@computer:~/example/python-example-2024$ docker run -it -v ~/example/model:/challenge/model -v ~/example/test_data:/challenge/test_data -v ~/example/test_outputs:/challenge/test_outputs -v ~/example/training_data:/challenge/training_data image bash

        root@[...]:/challenge# ls
            Dockerfile             README.md         test_outputs
            evaluate_model.py      requirements.txt  training_data
            helper_code.py         team_code.py      train_model.py
            LICENSE                run_model.py      [...]

        root@[...]:/challenge# python train_model.py -d training_data -m model -v

        root@[...]:/challenge# python run_model.py -d test_data -m model -o test_outputs -v

        root@[...]:/challenge# python evaluate_model.py -d test_data -o test_outputs
        [...]

        root@[...]:/challenge# exit
        Exit

## What else do I need?

This repository does not include data or the code for generating ECG images. Please see the above instructions for how to download and prepare the data.

This repository does not include code for evaluating your entry. Please see the [evaluation code repository](https://github.com/physionetchallenges/evaluation-2024) for code and instructions for evaluating your entry using the Challenge scoring metric.

## How do I learn more? How do I share more?

Please see the [Challenge website](https://physionetchallenges.org/2024/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges). Please do not make pull requests, which may share information about your approach.

## Useful links

* [Challenge website](https://physionetchallenges.org/2024/)
* [MATLAB example code](https://github.com/physionetchallenges/matlab-example-2024)
* [Evaluation code](https://github.com/physionetchallenges/evaluation-2024)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2024/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)
