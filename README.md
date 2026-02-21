# 👁️ See4You

**See4You** is an **Image Captioning** project developed with the core purpose of **assisting visually impaired individuals**. The system processes environmental images and describes the scene in natural language, promoting greater autonomy and digital inclusion.

---

## ⚙️ Architecture and Performance

To ensure the project can run on resource-constrained devices (such as smartphones or embedded assistive systems), computational efficiency was the top priority.

The final model utilizes the following architecture:
* **Encoder (Vision):** **MobileNetV3** — A pre-trained convolutional neural network responsible for extracting the image's vector representation.
* **Decoder (Language):** **GRU** (Gated Recurrent Unit) — A recurrent neural network responsible for text generation.

### Why this choice?

We conducted rigorous testing comparing different recurrent networks and pre-trained convolutional networks. The **MobileNetV3 + GRU** combination achieved metrics close to other architectures but with a significant reduction in execution time.

| Architecture Comparison | Speed Gain |
| :--- | :--- |
| **vs. MobileNetV3 + LSTM** | ⚡ **2.0x faster** |
| **vs. ResNet50 + GRU** | ⚡⚡ **2.5x faster** |

This translates to lower latency between image capture and the auditory description for the user—a critical factor for accessibility applications.

---

## 🛠️ Installation and Setup

The project is structured to be reproducible and easy to configure. Follow the steps below to set up your environment and train the model.

### 1. Clone and Install Dependencies

Clone this repository and install the required libraries:

```bash
git clone [https://github.com/your-user/see4you.git](https://github.com/your-user/see4you.git)
cd see4you
pip install -r requirements.txt
```

### 📥 2. Data Download

Before starting the training, you must configure the environment and download the necessary data. Run the **`setup.ipynb`** notebook to perform this process.

**What this notebook does:**
* **Dataset:** Downloads and extracts the image and caption dataset.
* **Embeddings:** Downloads the **FastText** pre-trained embeddings.
* **Structure:** Automatically creates the `/data` and `/embeddings` folders in the project root.

### 🔬 3. Data Analysis and Treatment (EDA)

Next, you need to process the data used for training. Run the **`eda.ipynb`** notebook to complete this step.

**What this notebook does:**
* **Exploratory Data Analysis:** Generates statistics and visualizations regarding images and caption lengths.
* **Cleaning:** Applies filters and processing to remove noise or inconsistent data.
* **Export:** Saves the cleaned dataset in the **`data/cleaned`** folder, which will be the source for training.

### 📊 4. Training and Evaluation

With the data organized, run the **`training.ipynb`** notebook to start the pipeline.

**The execution flow includes:**
1. **Preprocessing:** Loading DataLoaders and tokenization.
2. **Modeling:** Instantiating the **MobileNetV3 + GRU** architecture.
3. **Training:** Running training epochs while monitoring the *Loss*.
4. **Testing:** Automatic evaluation using similarity metrics on the test set.
