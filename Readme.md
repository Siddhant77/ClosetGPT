
## 🛠️ Installation

```bash
conda create -n outfit-transformer python=3.12.4
conda activate outfit-transformer
conda env update -f environment.yml
```

## 📥 Download Datasets & Checkpoints

```bash
mkdir -p datasets
gdown --id 1ox8GFHG8iMs64iiwITQhJ47dkQ0Q7SBu -O polyvore.zip
unzip polyvore.zip -d ./datasets/polyvore
rm polyvore.zip

mkdir -p checkpoints
gdown --id 1mzNqGBmd8UjVJjKwVa5GdGYHKutZKSSi -O checkpoints.zip
unzip checkpoints.zip -d ./checkpoints
rm checkpoints.zip
```
## Demo

Download the compatibility model, the complementry model and the precomputed embeddings from the shared google drive [link] (https://drive.google.com/drive/folders/19xzBp3dP33lMWMaNYkKFgpePmPBjb6jm?usp=share_link). Download the folders to ./closetGPT

Follow the steps below to run the demo:

#### Build Database (skip)
```
python -m src.demo.1_generate_rec_embeddings \
--checkpoint $PATH/OF/MODEL/.PT/FILE
```

#### Build Faiss Index. (skip)
```
python -m src.demo.2_build_index
```

#### Run Demo
```
python -m src.demo.3_run
```

## ⚠️ Note

This is a non-official implementation of the Outfit Transformer model. The official repository has not been released yet.
Our work is intened to build upon and improve the work done by [outfit-transformer](https://github.com/owj0421/outfit-transformer). 

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

If you use this repository, please mention the original GitHub repository by linking to [outfit-transformer](https://github.com/owj0421/outfit-transformer). This helps support the project and acknowledges the contributors.