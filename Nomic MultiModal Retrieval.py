# Databricks notebook source
# MAGIC %md
# MAGIC #Overview
# MAGIC
# MAGIC This demo showcases how to do image retrieval with text using Databricks' Vector Search. By the end of this demo, you will have created a vector search index containing image embeddings from the COCO dataset and be able to retrieve said images by asking/submitting text. 
# MAGIC
# MAGIC The demo will use the following resources:
# MAGIC 1. COCO 2017 Image dataset 
# MAGIC 2. Nomic-embed-text-v1.5 and Nomic-embed-vision-v1.5 embedding models
# MAGIC 3. Databricks Vector Search
# MAGIC
# MAGIC **Did you already run Step 1 to 3?** You can skip to Step 4 to use it without issue! (still need to install the dependencies)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #Summary
# MAGIC
# MAGIC When retrieving content in various types of data, you need to use multi-modal embedding models. This is because the embeddings need to be aligned to the same embeddings space. This is why a text embedding model like GTE cannot be used for images and it has been trained purely for text. 
# MAGIC
# MAGIC Nomic aligned their vision embedding model with their text model allowing us to use their embedding models to embed both our text and images. We must use their two image and vision embedding models to maintain that shared embedding space 
# MAGIC
# MAGIC The demo will use the nomic-embed-vision-v1.5 model to convert images from the COCO dataset into embeddings. We will store these embeddings into a Delta table, convert the table into a vector search index and sync it with a vector search endpoint. 
# MAGIC
# MAGIC To query the index, we need to convert our text request into embeddings and then submit those embeddings into the vector search. This is where we use nomic-embed-text-v1.5 but we don't store the text embeddings. This text is only used to retrieve the images we are looking for so it is not necessary to save these embeddings for this demo

# COMMAND ----------

# MAGIC %md
# MAGIC #Step 0: Install Dependencies
# MAGIC
# MAGIC Run the correct corresponding cell depending on your compute type
# MAGIC
# MAGIC DBR ML 16.x+: Run Cell 2
# MAGIC
# MAGIC Serverless Compute: Run Cell 3

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-vectorsearch transformers pillow matplotlib gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-vectorsearch torch transformers pillow matplotlib einops
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import requests
import zipfile
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from databricks.vector_search.client import VectorSearchClient
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor
from pyspark.sql.types import ArrayType, FloatType, StringType, StructType, StructField
from pyspark.sql.functions import input_file_name, regexp_extract, regexp_replace, col



# COMMAND ----------

# MAGIC %md
# MAGIC ### Change these to the locations of your choice or in the widgets above

# COMMAND ----------

dbutils.widgets.text("catalog_name", "austin_choi_demo_catalog", "Data UC Catalog")
dbutils.widgets.text("schema_name", "demo_data", "Data UC Schema")
dbutils.widgets.text("table_name", "image_data_table", "Data UC Table")
dbutils.widgets.text("volume_name", "image_data", "Data Volume Table")
dbutils.widgets.text("embedding_table_name", "image_data_embedding", "Data Embedding Table")
dbutils.widgets.text("index_name", "image_data_index", "Data Index Table")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Change the catalog/schema/table you want to use for this demo in the widget or below

# COMMAND ----------

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
image_table_name = dbutils.widgets.get("table_name")
embedding_table_name = dbutils.widgets.get("embedding_table_name")
index_name = dbutils.widgets.get("index_name")
volume_name = dbutils.widgets.get("volume_name")
vector_search_endpoint_name = "one-env-shared-endpoint-4" # Make sure you have one. Go to Compute to create one if you do not have one

# COMMAND ----------

spark.sql(
f"""
    CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}
"""
)

# COMMAND ----------

spark.sql(
f"""
    CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.{volume_name}
"""
)

# COMMAND ----------

# MAGIC %md
# MAGIC #Step 1: Download the Images
# MAGIC
# MAGIC It's largely up to you on what you would like to download since the COCO dataset is well-labeled as well. For the purpose of this demo, we wanted to highlight how pictures with no labels or other metadata can be retrieved using text. So, we download some images unlabeled. 
# MAGIC
# MAGIC The validation set contains 5K images, a small subset of the hundreds of thousands the dataset has

# COMMAND ----------

# DBTITLE 1,Download and save to a volume
dest_dir = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(destination, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))

# Download validation images
val_url = "http://images.cocodataset.org/zips/val2017.zip"
val_zip = os.path.join(dest_dir, "val2017.zip")
download_file(val_url, val_zip)

# Extract files
with zipfile.ZipFile(val_zip, 'r') as zip_ref:
    zip_ref.extractall(dest_dir)

# Clean up zip files
os.remove(val_zip)

print(f"COCO 2017 validation dataset downloaded to {dest_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now we grab the files from volumes, do some clean up and save them to a delta table

# COMMAND ----------

dest_dir = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/val2017"
image_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").load(dest_dir)
image_df = image_df.withColumn('path', regexp_replace(col('path'), '^dbfs:', ''))
image_df.write.mode('overwrite').saveAsTable(f"{catalog_name}.{schema_name}.{image_table_name}")

image_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC #Step 2: Download the Models
# MAGIC
# MAGIC We will use Huggingface's transformer library to quickly download the nomic embedding models. Even though their vision and text models are distinct repos on huggingface, they still share the same embedding space so we can use them for our multimodal RAG retrieval. However, we still need to use the correct model to embed images and text

# COMMAND ----------

# DBTITLE 1,Check to see if we are using GPUs and use it if we are
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###We use the transformer library to quickly download the model from huggingface

# COMMAND ----------

# DBTITLE 1,Download the Model
processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### This function will help generate the embeddings based on the code provided in the model card
# MAGIC
# MAGIC See the code here: https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5
# MAGIC

# COMMAND ----------

# DBTITLE 1,Helper Function to generate embeddings
def generate_image_embedding(image_path):
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = vision_model(**inputs)
        embedding = outputs.last_hidden_state
        normalized = F.normalize(embedding[:, 0], p=2, dim=1)
        return normalized
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pull the file paths and generate the embeddings

# COMMAND ----------

# DBTITLE 1,Pull the file paths
# We don't need to run all 5000 for this demo. Coco has tens of thousands of images but this demo would take ages to run if we tried to do all of them
image_df = spark.table(f"{catalog_name}.{schema_name}.{image_table_name}").limit(500) 

# Collect paths to process - if dataset is large, consider using more efficient approaches
image_paths = image_df.select("path").collect()
path_list = [row["path"] for row in image_paths]

# COMMAND ----------

# DBTITLE 1,Generate Embeddings
image_data = []

#Production Tip: Serve the embedding model instead and use AI Query for batch infernece. This is not efficient if there are thousands of images
for i, image_path in enumerate(path_list):
    image_id = i
    embedding = generate_image_embedding(image_path)
    if embedding is not None:
        flatten_embedding = embedding.flatten()
        image_data.append({
            "image_id": image_id,
            "filepath": image_path,
            "embedding": flatten_embedding,
        })

print(f"Generated embeddings for {len(image_data)} images")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the generated embeddings to a Delta Table and enable Change Data Feed for vector search

# COMMAND ----------

# DBTITLE 1,Save the Embeddings
import pandas as pd


# Create a DataFrame with image data
image_df_new = pd.DataFrame([{
    "image_id": item["image_id"],
    "filepath": item["filepath"],
    "embedding": item["embedding"].tolist()
} for item in image_data])

# Convert pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(image_df_new)

# Save the DataFrame as a Delta table
delta_table_path = f"{catalog_name}.{schema_name}.{embedding_table_name}"
spark_df.write.format("delta").mode("overwrite").saveAsTable(delta_table_path)


# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE identifier(CONCAT(:catalog_name||'.'||:schema_name||'.'||:embedding_table_name)) SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md
# MAGIC #Step 3: Set up the Vector Search Index
# MAGIC
# MAGIC Now that we have our image embeddings, we need to put them into a vector search index to performan a vector search. We can programatically create them below or use the UI to do this. 
# MAGIC
# MAGIC In the UI, you can navigate to the delta table you created above that contains the embeddings, and click create vector search index
# MAGIC
# MAGIC Once you complete this step, you can always come back and go straight to step 4 to demo the vector search endpoint. 

# COMMAND ----------

# Create a Vector Search Index
vs_client = VectorSearchClient()

# Define the endpoint configuration
endpoint_name = vector_search_endpoint_name
delta_table_path = f"{catalog_name}.{schema_name}.{embedding_table_name}"
delta_index_name = f"{catalog_name}.{schema_name}.{index_name}"

# COMMAND ----------

# Create the vector search index

# vs_client.create_delta_sync_index(
#     index_name=delta_index_name,
#     endpoint_name=endpoint_name,
#     pipeline_type="TRIGGERED",
#     primary_key="image_id",
#     embedding_vector_column="embedding",
#     source_table_name=delta_table_path,
#     embedding_dimension=len(image_data[0]["embedding"])
# )

vs_client.create_delta_sync_index_and_wait(
    index_name=delta_index_name,
    endpoint_name=endpoint_name,
    pipeline_type="TRIGGERED",
    primary_key="image_id",
    embedding_vector_column="embedding",
    source_table_name=delta_table_path,
    embedding_dimension=len(image_data[0]["embedding"])
)


print(f"Vector search index '{index_name}' created on endpoint '{endpoint_name}'")

# COMMAND ----------

# MAGIC %md
# MAGIC #Step 4: Set up the Text Embedding model
# MAGIC
# MAGIC Now we need to download the text embedding model to convert our text query into embeddings. We follow a similar process to download the text embedding model. Then, we set up helper functions to query the vector search index

# COMMAND ----------

# Create a Vector Search Index
vs_client = VectorSearchClient()

# Define the endpoint configuration
endpoint_name = vector_search_endpoint_name
delta_table_path = f"{catalog_name}.{schema_name}.{embedding_table_name}"
delta_index_name = f"{catalog_name}.{schema_name}.{index_name}"

# COMMAND ----------

from transformers import AutoModel, AutoTokenizer

# Load the text model and tokenizer
text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
text_model.eval()
text_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Functions to generate the embeddings. This code is also available on the model card 
# MAGIC
# MAGIC Huggingface model card: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

# COMMAND ----------

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_text_embedding(query_text):
    encoded_input = text_tokenizer(question, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
      model_output = text_model(**encoded_input)
    text_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    text_embeddings = F.layer_norm(text_embeddings, normalized_shape=(text_embeddings.shape[1],))
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    text_flatten_embedding = text_embeddings.flatten()
    return text_flatten_embedding.tolist()
  
def search_images_by_text(query_text, top_k=5):
    # Generate embedding for the query text
    query_embedding = generate_text_embedding(query_text)

    index = vs_client.get_index(endpoint_name=vector_search_endpoint_name, index_name=delta_index_name)

    # If you want to use hybrid search, uncomment the line below
    # results = index.similarity_search(num_results=3, columns=["image_id", "filepath"], query_text=query_text, query_vector=query_embedding, query_type="hybrid")
    results = index.similarity_search(num_results=5, columns=["image_id", "filepath"], query_vector=query_embedding)
    
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC #Step 5: Try it!
# MAGIC
# MAGIC We are ready to query the images! 
# MAGIC
# MAGIC One caveat: The nomic models expect a **task instruction prefix** before the query. Some examples are `search_document: <your text here>` and `clustering: <your text here>`. You must add this prefix for accurate retrieval. Since we are trying to do a search on images, we must provide `search_query: <your text here>`.

# COMMAND ----------

question = 'search_query: something foresty with people'
search_results = search_images_by_text(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ### We will use matplotlib plt to construct the images

# COMMAND ----------

import matplotlib.pyplot as plt

#grab the file paths from the vector search results
file_paths = [path[1] for path in search_results['result']['data_array']]

# Display a few sample images
fig, axes = plt.subplots(1, len(file_paths), figsize=(15, 5))
for i, path in enumerate(file_paths):
    img = Image.open(path)
    axes[i].imshow(img)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Not bad! 
# MAGIC
# MAGIC You can try different text query and see what images it tends to pull! You can also adjust how many images the vector search endpoint pull by adjusting `num_result` on the vector search endpoint

# COMMAND ----------

import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt2
from io import BytesIO

# Add this function to create a compatible scope with your notebook
def search_and_display_images(question):
    """
    Performs image search using the exact pattern from the notebook
    """
    # Use the exact pattern from your original code
    print("running the actual function")
    search_results2 = search_images_by_text(question)
    
    # Grab the file paths from the vector search results
    file_paths = [path[1] for path in search_results2['result']['data_array']]
    
    # Create a figure with the images using matplotlib (just like in your notebook)
    fig, axes = plt2.subplots(1, len(file_paths), figsize=(15, 5))
    
    # Handle case with only one image
    if len(file_paths) == 1:
        img = Image.open(file_paths[0])
        axes.imshow(img)
        axes.set_title(f"Image 1")
        axes.axis('off')
    else:
        # Multiple images
        for i, path in enumerate(file_paths):
            img = Image.open(path)
            axes[i].imshow(img)
            axes[i].set_title(f"Image {i+1}")
            axes[i].axis('off')
    
    plt2.tight_layout()
    
    # Save the figure to a bytes buffer
    buf = BytesIO()
    plt2.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to PIL Image for Gradio
    plot_img = Image.open(buf)
    
    # Close the figure to free memory
    plt2.close(fig)
    
    return plot_img

# Simple Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Search Interface")
    
    with gr.Row():
        # Input textbox
        query_input = gr.Textbox(
            label="Search Query",
            placeholder="Enter search query (e.g., search_query: people having fun outside)",
            value="search_query: people having fun outside"  # Set default value like in your notebook
        )
    
    with gr.Row():
        # Button to trigger search
        search_button = gr.Button("Search Images")
    
    with gr.Row():
        # Output image (will show the matplotlib figure with all images)
        output_image = gr.Image(label="Search Results", type="pil")
    
    # Example queries
    gr.Examples(
        examples=[
            "search_query: people having fun outside", 
            "search_query: dogs in the park", 
            "search_query: sunset beach"
        ],
        inputs=query_input
    )
    
    # Connect the button to the search function
    search_button.click(
        fn=search_and_display_images,
        inputs=[query_input],
        outputs=[output_image]
    )
    
    # Also allow pressing Enter to search
    query_input.submit(
        fn=search_and_display_images,
        inputs=[query_input],
        outputs=[output_image]
    )

demo.launch(share=True)

# COMMAND ----------

