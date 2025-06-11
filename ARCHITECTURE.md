### Project Overview

This project is an unofficial implementation of the "Outfit Transformer," a deep learning model designed to understand fashion and provide outfit recommendations. The core functionality includes:

*   **Outfit Compatibility**: Assessing whether a set of items form a coherent and fashionable outfit.
*   **Complementary Item Recommendation**: Suggesting items that would complement a given item or set of items.

The project also includes extended functionality for creating personalized outfit recommendations based on weather and travel itineraries.

### Technology Stack

The project is built on a modern Python-based stack, including:

*   **Deep Learning**: `torch`, `transformers`, `clip`
*   **Vector Search**: `faiss-cpu`
*   **Web Framework**: `fastapi`, `gradio`
*   **Data Handling**: `pandas`, `numpy`

### Architecture

The project follows a modular architecture, with a clear separation of concerns. The main components are:

*   **Data**: The `src/data` directory contains all the code for loading, preprocessing, and augmenting the data. It includes custom `Dataset` classes for the Polyvore dataset.
*   **Models**: The `src/models` directory contains the core deep learning models. The main model is the `OutfitTransformer`, which is implemented in `src/models/outfit_transformer.py`. The `outfit_clip_transformer.py` file contains a version of the model that uses the CLIP model for multimodal (image and text) understanding. The `src/models/modules` directory contains the building blocks of the models, including encoders for images and text.
*   **Training and Evaluation**: The `src/run` directory contains scripts for training and evaluating the models. The `src/evaluation` directory contains the code for computing the evaluation metrics.
*   **Demo**: The `src/demo` directory contains the code for the interactive demo. The demo uses `fastapi` to create a web server and `gradio` to create the user interface.
*   **Utilities**: The `src/utils` directory contains utility functions that are used throughout the project.

### Architectural Diagram

```mermaid
graph TD
    A[Data] --> B(Models);
    B --> C{Training & Evaluation};
    B --> D[Demo];
    E[Utilities] --> B;
    E --> C;
    E --> D;

    subgraph Data
        direction LR
        A1[Polyvore Dataset] --> A2[Dataloaders];
    end

    subgraph Models
        direction LR
        B1[Outfit Transformer] --> B2[Encoders];
        B2 --> B3[Image Encoder];
        B2 --> B4[Text Encoder];
    end

    subgraph "Training & Evaluation"
        direction LR
        C1[Training Scripts] --> C2[Evaluation Metrics];
    end

    subgraph Demo
        direction LR
        D1[FastAPI Server] --> D2[Gradio UI];
    end
