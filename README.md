# code project of our final project

This README file how to use our self-supervised model and how to reproduce our result

# use our model directly through colab
The gec.ipynb provides a user friendly colab version of use our SFT llama2 GEC models.

# reproduce paper result

## Setting Up the Environment

### Requirements
- Python 3.10
- Linux environment

### Preparing the Environment
To get started, create a new virtual environment and install the necessary dependencies:

```bash
# Create a Python 3.10 virtual environment
virtualenv -p python3.10 llama2_env
source llama2_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```
## prepare data
download the file from https://drive.google.com/file/d/1GrGif8EHg8vCxOb0M7DC3VOECHv7SkdB/view?usp=drive_link
```python
prepare_rlhfdataset.py
preparesft_dataset.py
```
## Training the SFT Model

To train the model for a grammar correction task, execute the following command:

```bash
python sft_train.py --model_name [Huggingface model path] --load_in_4bit --use_peft --batch_size 2 --gradient_accumulation_steps 1 --output_dir /sftmodels/[model name] --logging_steps 100 --lora_module llama Z
```

Make sure to replace `[Huggingface model path]` and `[model name]` with the appropriate values specific to your scenario.

## Merging Trained Models

After training, you can merge your models with the following command:

```bash
python merge.py --adapter_model_name /sftmodels/[model] --base_model_name /workspace/mpt-7b --output_name merged/mpt
```

Replace `[model]` with the name of your trained model.

## Generating SFT Results

Use the SFTgeneration script to generate results with the merged model:

```bash
python SFTgeneration.py --[merged model path]
```

The results will be stored in `test_data/sftresult`.

## Applying Reinforcement Learning from Human Feedback (RLHF)

For RLHF, employ the following command:

```bash
python rlhf.py --ppo_config.model_name [path to merged sft] --ppo_config.log_with wandb
```

Results will be available in `ppo.txt`.

## Running Additional Scripts

If necessary, the following script can be executed to generate zero shot prediction:

```bash
python falcon.py llama.py mpt.py
```

Don't forget to update the path to the downloaded Hugging Face model checkpoints.

## Evaluation

### Setting Up Python 2.7 Environment

The evaluation requires Python 2.7, so you must set up an additional virtual environment for it.

### Using M2Scorer

To evaluate your model, navigate to the `m2scorer` directory and use the following command:

```bash
python m2scorer prediction_text.txt test_data/standard/conll14st-test.tok.src
```

Make sure to adapt any placeholders like `[merged model path]` to the correct paths relevant to your setup.

This guide should provide you with all the necessary steps to utilize the Llama2 SFT model effectively for grammar corrections. Please follow each step carefully and refer to this guide as needed during your work.
````
