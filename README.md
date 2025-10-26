# VGG Neural Network Experiment Framework

A comprehensive framework for experimenting with VGG neural network architectures on CIFAR-10 dataset using Weights & Biases (wandb) for hyperparameter optimization and experiment tracking.


### Prerequisites

1. **Python Installation**
   - Ensure Python  3.11.9+ is installed on your system
   - Check your Python version:
     ```bash
     python --version
     ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **GPU Setup (Optional but Recommended)**
   If you have a CUDA-compatible GPU, uninstall the cpu version of pytorch installed in step 2 and install PyTorch with CUDA support for faster training:
   To uninstall
   ```bash
   pip uninstall -y torch torchvision  
    ```
    
   ```bash
   # For CUDA 12.6
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   
   # Check GPU availability
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```
   
   **Note:** Visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for specific CUDA version compatibility.

4. **Weights & Biases Setup**
   - Create a free account at [wandb.ai](https://wandb.ai)
   - Get your API key from [wandb.ai/settings](https://wandb.ai/settings)
   - When running for the first time, wandb will prompt you to enter your API key

## Sweep Configuration

This project includes a **`config.json`** file that defines the hyperparameter search space for automated experimentation.

## Usage Workflow

### Step 1: Register a Hyperparameter Sweep

```bash
wandb sweep config.json -e <username> -p <project_name>
```

**Example:**
```bash
wandb sweep config.json -e krishnaprasad-student -p wandb_vgg_experiments_4 
```

**Output:** This command will return a **Sweep ID** (e.g., `ug2uxgsu`). **Copy and save this ID** - you'll need it for the next step.

### Step 2: Start Training Agents

```bash
python trainer.py --sweep_id=<SWEEP_ID> --project=<PROJECT_NAME> --count=<NUM_RUNS>
```

**Example:**
```bash
python trainer.py --sweep_id=62z9brzv --project=wandb_vgg_experiments_4 --count=25
```

**Parameters:**
- `--sweep_id`: The sweep ID from Step 1
- `--project`: Your wandb project name
- `--count`: Number of experiments to run (default: 5)

### Step 3: Download Best Model

1. **Open the Jupyter Notebook:**
   ```bash
   download_model.ipynb
   ```

2. **Run the notebook and provide below details at the beginning of file:**
   - **Username**: Username of wandb.io account
   - **Project Name**: Project name
   - **Run ID**: Needed only if Option B (manual download option) chosen. Get this from your wandb dashboard like by identifying the run which has maximum validation accuracy (best performing run) Ex: hi5mjzjd
   - The notebook will download the previously trained model from wandb.io and **print the local file path**


   ### Option A: Automatically select best model (based on validation accuracy) -By default
         run = get_best_run(username, project_name)

   Only Username and Project Name are required. Run Id if any provided will be ignored. This will fetch the run with the highest validation accuracy automatically and download the corresponding model.

   ### Option B: to download model using specific Run ID
   Comment the line above in option A and uncomment below line instead in download_model.ipynb file. Also provide valid run id at the beginning

         run = get_run(username, project_name, run_id)

### Step 4: Test the Model

Seed configuration has been set in config.json file. "seed": { "value": 42 }

```bash
python tester.py <path_to_model>
```

**Example:**
```bash
python tester.py .\artifacts\vgg-v16\model.pth
```

**Output:** The script will load the model, test it on CIFAR-10 test data, and print:
- Accuracy
- Loss

