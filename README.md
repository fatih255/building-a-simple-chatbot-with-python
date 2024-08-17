# Building a Simple Chatbot using Deep Learning

In this tutorial, we will walk through the process of building a simple chatbot using deep learning techniques. This project utilizes TensorFlow and Keras.

You can read the full article here: [Building a Simple Chatbot using Deep Learning](https://medium.com/@fatih969692/building-a-simple-chatbot-using-deep-learning-2030cf58a231)

article 2 (for app-csv.py version): https://medium.com/@fatih969692/building-a-sophisticated-faq-chatbot-leveraging-tensorflow-fuzzy-matching-and-glove-embeddings-bc08a6569d64

## Installation Steps

1. Activate the virtual environment:

    ```bash
    . .venv/bin/activate
    ```

2. Navigate to the virtual environment directory:

    ```bash
    cd .venv/
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:

    ```bash
    python app.py
    ```

### Running with `app-csv.py`

To use the `app-csv.py` version, you need to download and add the `glove.6B.100d.txt` file to the `data/` directory.

- Download the file from [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/).
- Place the file in `data/glove.6B.100d.txt`.

After setting up the file, you can run the application with:

```bash
python app-csv.py
