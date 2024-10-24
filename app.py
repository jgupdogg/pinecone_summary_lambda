import os
import json
import datetime
import logging
import openai
from pinecone import Pinecone  # Ensure you're using the correct Pinecone client library

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def format_date(date_obj):
    return date_obj.strftime('%Y-%m-%d')

def generate_date_range(start_date, end_date):
    dates = []
    delta = datetime.timedelta(days=1)
    while start_date <= end_date:
        dates.append(format_date(start_date))
        start_date += delta
    return {"$in": dates}

def is_valid_vector(vector, dimension=1536):
    return (
        isinstance(vector, list) and 
        len(vector) == dimension and 
        all(isinstance(item, (int, float)) for item in vector)
    )

def clean_empty_arrays_and_objects(obj):
    keys_to_delete = []
    for key, value in obj.items():
        if isinstance(value, dict):
            clean_empty_arrays_and_objects(value)
            if not value:
                keys_to_delete.append(key)
        elif isinstance(value, list) and not value:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del obj[key]

def initialize_pinecone():
    # Retrieve Pinecone API key from environment variables
    pinecone_api_key = os.environ.get("pineconeKey")
    
    # Validate API key
    if not pinecone_api_key:
        logger.error("Pinecone API key is missing in environment variables.")
        raise ValueError("Pinecone API key is missing.")
    
    logger.info("Pinecone API Key retrieved successfully.")
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        # Connect to the 'agent-alpha' index
        pinecone_index = pc.Index('agent-alpha')  
        logger.info("Pinecone initialized successfully and connected to 'agent-alpha' index.")
        return pinecone_index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

def generate_recent_dates(days=5):
    dates = []
    for i in range(days):
        date_obj = datetime.datetime.now() - datetime.timedelta(days=i)
        dates.append(format_date(date_obj))
    return {"$in": dates}

def is_valid_vector(vector, dimension=1536):
    return (
        isinstance(vector, list) and 
        len(vector) == dimension and 
        all(isinstance(item, (int, float)) for item in vector)
    )

def create_dense_vector(input_text, model="text-embedding-3-small"):
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
        embedding = response['data'][0].embedding
        return embedding
    except Exception as e:
        logger.error(f"Error creating dense vector: {str(e)}")
        return None

def lambda_handler(event, context):
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Define CORS headers
    cors_headers = {
        "Access-Control-Allow-Origin": "*",  # Adjust as needed for security
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, apikey",
        "Content-Type": "application/json",
    }
    
    # Log the event for debugging
    logger.info(f"Event received: {json.dumps(event)}")
    
    # Extract the HTTP method from the event
    method = event.get('httpMethod')
    
    if not method:
        # For API Gateway HTTP API (payload version 2.0)
        if 'requestContext' in event and 'http' in event['requestContext']:
            method = event['requestContext']['http'].get('method')
        else:
            # Default to POST if method cannot be determined
            method = 'POST'
    
    # Handle OPTIONS method for CORS preflight
    if method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': ''
        }
    
    # Ensure the request method is POST
    if method != 'POST':
        return {
            'statusCode': 405,
            'headers': cors_headers,
            'body': json.dumps('Method Not Allowed')
        }
    
    try:
        # Initialize Pinecone and get the index
        pinecone_index = initialize_pinecone()
        
        # Set OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OpenAI API key is missing in environment variables.")
            raise ValueError("OpenAI API key is missing.")
        openai.api_key = openai_api_key
        
        # Extract the JSON payload from the request
        body = event.get('body', '{}')
        if event.get('isBase64Encoded', False):
            import base64
            body = base64.b64decode(body).decode('utf-8')
        payload = json.loads(body)
        logger.info(f"Received payload: {json.dumps(payload)}")
        
        # Extract required and optional fields
        symbol = payload.get('symbol')
        cat = payload.get('cat')
        significancescore = payload.get('significancescore')
        sentimentscore = payload.get('sentimentscore')
        start_date_str = payload.get('start_date')
        end_date_str = payload.get('end_date')
        search_string = payload.get('search_string', '').strip()
        top_k = payload.get('top_k', 3)
        # Default dimension for 'text-embedding-ada-002' model
        dimension = 1536
        
        # Generate dynamic filters for the query
        dynamic_filters = {}
        
        # Handle date range
        if start_date_str and end_date_str:
            try:
                start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
                end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
                if start_date > end_date:
                    raise ValueError("start_date cannot be after end_date.")
                dynamic_filters["created"] = generate_date_range(start_date, end_date)
            except ValueError as ve:
                logger.error(f"Invalid date format: {ve}")
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({'error': f'Invalid date format: {ve}'})
                }
        else:
            # Default to the last 5 days if dates are not provided
            dynamic_filters["created"] = generate_recent_dates(5)
        
        if symbol:
            dynamic_filters["symbol"] = {"$in": symbol if isinstance(symbol, list) else [symbol]}
        if cat:
            dynamic_filters["cat"] = {"$in": list(filter(None, cat))}
        if significancescore:
            dynamic_filters["significancescore"] = {"$in": significancescore if isinstance(significancescore, list) else [significancescore]}
        if sentimentscore:
            dynamic_filters["sentimentscore"] = {"$in": sentimentscore if isinstance(sentimentscore, list) else [sentimentscore]}
        
        # Clean the dynamic_filters object
        clean_empty_arrays_and_objects(dynamic_filters)
        
        # Handle 'search_string'
        if search_string:
            # Generate vector from search string
            vector = create_dense_vector(search_string)
            if not vector or not is_valid_vector(vector, dimension):
                return {
                    'statusCode': 500,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Error generating vector from search string.'})
                }
            logger.info(f"Generated vector of length {len(vector)} from search string.")
            # log the first few numbers of the vector
            logger.info(f"Vector: {vector[:5]}")
            
        else:
            # Create a zero-filled vector based on 'dimension'
            vector = [0.0] * dimension
            logger.info(f"No search string provided. Generated zero vector of dimension {dimension}.")
        
        # Perform a query to the 'summaries' namespace
        try:
            query_results = pinecone_index.query(
                namespace="summaries",
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                # filter=dynamic_filters  # Apply dynamic filters if necessary
            )
            
            # Convert QueryResponse to a serializable dictionary
            serializable_results = {
                "matches": [
                    {
                        "id": match.id,
                        "score": match.score,
                        # Optionally include values and metadata
                        # "values": match.values,
                        "metadata": match.metadata
                    }
                    for match in query_results.matches
                ],
                "namespace": query_results.namespace,
            }
            
            logger.info(f"Query Results for 'summaries': {json.dumps(serializable_results)}")
        except Exception as e:
            logger.error(f"Error querying 'summaries' namespace: {str(e)}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'error': f"Error querying 'summaries' namespace: {str(e)}"})
            }
        
        # Prepare the combined results (only one namespace)
        combined_results = {
            "summaries": serializable_results
        }
        
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps(combined_results)
        }
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON payload.")
        return {
            'statusCode': 400,
            'headers': cors_headers,
            'body': json.dumps({'error': 'Invalid JSON payload.'})
        }
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps('Internal Server Error')
        }
