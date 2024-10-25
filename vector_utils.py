import logging
import openai

logger = logging.getLogger()

def is_valid_vector(vector, dimension=1536):
    return (
        isinstance(vector, list) and 
        len(vector) == dimension and 
        all(isinstance(item, (int, float)) for item in vector)
    )

def create_dense_vector(input_text, model="text-embedding-ada-002"):
    """
    Creates an embedding for the given input text using OpenAI's embedding model.

    Parameters:
    input_text (str or list): The text (or list of texts) to embed.
    model (str): The model to use for creating embeddings.

    Returns:
    list: The embedding vector.
    """
    try:
        # Send the embedding request to the API
        response = openai.Embedding.create(
            input=input_text,
            model=model
        )
        # Extract the embedding vector from the response
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        logger.error(f"Error creating dense vector: {str(e)}")
        return None
