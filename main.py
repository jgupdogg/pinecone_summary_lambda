import os
import json
import datetime
import logging
import base64
import openai
import asyncio

from data_cleaning import clean_empty_arrays_and_objects
from date_utils import generate_date_range, generate_recent_dates
from vector_utils import is_valid_vector, create_dense_vector
from db_utils import initialize_pinecone, get_article_info_from_snowflake, create_snowflake_session

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def lambda_handler(event, context):
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
    method = event.get('httpMethod') or event.get('requestContext', {}).get('http', {}).get('method', 'POST')
    
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
        
        # Create Snowflake session
        snowflake_session = create_snowflake_session()
        
        # Set OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OpenAI API key is missing in environment variables.")
            raise ValueError("OpenAI API key is missing.")
        openai.api_key = openai_api_key
        
        # Extract the JSON payload from the request
        body = event.get('body', '{}')
        if event.get('isBase64Encoded', False):
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
            # Log the first few numbers of the vector
            logger.info(f"Vector (first 5 elements): {vector[:5]}")
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
            
            # Log the entire query_results for debugging
            logger.debug(f"Full query_results: {json.dumps(query_results, default=str)}")
            
            article_ids_set = set()
            for match in query_results['matches']:
                metadata = match.get('metadata', {})
                article_ids_str = metadata.get('article_ids', '')
                if article_ids_str:
                    # Split the comma-separated string and strip whitespace
                    article_ids = [aid.strip() for aid in article_ids_str.split(',') if aid.strip()]
                    article_ids_set.update(article_ids)
                    logger.debug(f"Extracted article_ids: {article_ids}")
                else:
                    logger.warning("No 'article_ids' found in match metadata.")
            logger.info(f"Total unique article_ids extracted: {len(article_ids_set)}")

            if not article_ids_set:
                logger.warning("No article_ids found in any of the matches.")
            
            # Fetch site and url information from Snowflake
            article_info = get_article_info_from_snowflake(article_ids_set, snowflake_session) if article_ids_set else {}
            logger.info(f"Fetched article info: {article_info}")

            # Convert QueryResponse to a serializable dictionary and modify 'article_ids' in metadata
            serializable_results = {
                "matches": [],
                "namespace": query_results['namespace'],
            }

            for match in query_results['matches']:
                match_id = match['id']
                score = match['score']
                metadata = match.get('metadata', {})
                article_ids_str = metadata.get('article_ids', '')
                if article_ids_str:
                    article_ids = [aid.strip() for aid in article_ids_str.split(',') if aid.strip()]
                    # Build a dict mapping article_id to its site and url
                    articles_data = {}
                    for aid in article_ids:
                        if aid in article_info:
                            articles_data[aid] = article_info[aid]
                        else:
                            articles_data[aid] = {'site': None, 'url': None}
                    # Replace 'article_ids' in metadata with articles_data
                    metadata['article_ids'] = articles_data
                else:
                    metadata['article_ids'] = {}

                # Append the match to serializable_results
                serializable_results['matches'].append({
                    "id": match_id,
                    "score": score,
                    "metadata": metadata
                })

            # Prepare the response body
            response_body = {
                "summaries": serializable_results
            }

            # Log the response body
            logger.info(f"Response Body: {json.dumps(response_body)}")

        except Exception as e:
            logger.error(f"Error querying 'summaries' namespace: {str(e)}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'error': f"Error querying 'summaries' namespace: {str(e)}"})
            }
        
        # Close the Snowflake session
        try:
            snowflake_session.close()
            logger.info("Snowflake session closed successfully.")
        except Exception as e:
            logger.error(f"Error closing Snowflake session: {str(e)}")
        
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps(response_body)
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
