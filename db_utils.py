import os
import logging
from snowflake.snowpark import Session
from pinecone import Pinecone
from snowflake.snowpark.functions import col
import requests


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def initialize_pinecone():
    # Retrieve Pinecone API key from environment variables
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    # Validate API key
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY is missing in environment variables.")
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



def create_snowflake_session():
    """
    Establishes and returns a Snowflake Snowpark Session.
    """
   
    try:
        connection_parameters = {
            "account": f"{os.getenv('SNOWFLAKE_ACCOUNT')}.{os.getenv('SNOWFLAKE_REGION')}",
            "user": os.getenv('SNOWFLAKE_USER'),
            "password": os.getenv('SNOWFLAKE_PASSWORD'),
            "role": os.getenv('SNOWFLAKE_ROLE'),
            "warehouse": os.getenv('SNOWFLAKE_WAREHOUSE'),
            "database": os.getenv('SNOWFLAKE_DATABASE'),
            "schema": os.getenv('SNOWFLAKE_SCHEMA'),
        }
        
        # Create Snowpark session
        logger.info(f"Connection parameters: {connection_parameters}")
        # Establish a session with Snowflake
        logger.info("Establishing Snowflake session...")
        snowflake_session = Session.builder.configs(connection_parameters).create()
        logger.info("Connection to Snowflake successful!")
        return snowflake_session
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {str(e)}")
        raise




# db_utils.py

def get_article_info_from_snowflake(article_ids, session: Session):
    """
    Fetches site and URL information for the given article_ids from Snowflake without using pandas.
    
    Parameters:
    - article_ids (str, set, or list): A set or list of article IDs.
    - session (Session): An active Snowflake session.
    
    Returns:
    - dict: A dictionary mapping each article_id to its corresponding site and URL.
    """
    try:
        if isinstance(article_ids, (set, list)):
            article_ids_list = [aid.strip() for aid in article_ids]
        elif isinstance(article_ids, str):
            article_ids_list = [aid.strip() for aid in article_ids.split(',') if aid.strip()]
        else:
            logger.error("article_ids must be a string, set, or list.")
            return {}

        # Remove any empty strings from the list
        article_ids_list = [aid for aid in article_ids_list if aid]

        # Log the article IDs being queried
        logger.info(f"Searching for article_ids: {article_ids_list}")
        
        # Construct the SQL query with placeholders
        placeholders = ', '.join([f"'{aid}'" for aid in article_ids_list])
        query = f"""
        SELECT ID, SITE, URL
        FROM STOCK_NEWS
        WHERE ID IN ({placeholders})
        """
        
        logger.debug(f"Executing query: {query}")
        
        # Execute the query and fetch results
        result = session.sql(query).collect()
        logger.info(f"Fetched {len(result)} rows from Snowflake.")
        
        # Build the dictionary
        article_info = {}
        for row in result:
            article_id = row['ID']
            site = row['SITE']
            url = row['URL']
            article_info[article_id] = {'site': site, 'url': url}
        
        logger.info(f"Constructed article_info dictionary with {len(article_info)} entries.")
        return article_info

    except Exception as e:
        logger.error(f"Error fetching article info from Snowflake: {e}")
        return {}
    

def get_quote_info(symbols):
    """
    Fetches quote information for the given list of symbols from the Financial Modeling Prep API.

    Parameters:
    - symbols (list): A list of stock symbols.

    Returns:
    - dict: A dictionary mapping each symbol to its quote data.
    """
    try:
        fmp_api_key = os.environ.get("FMP_API_KEY")
        if not fmp_api_key:
            logger.error("FMP_API_KEY is missing in environment variables.")
            raise ValueError("Financial Modeling Prep API key is missing.")

        symbols_str = ",".join(symbols)
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbols_str}?apikey={fmp_api_key}"

        logger.info(f"Fetching quote data from FMP API for symbols: {symbols_str}")
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad requests (4xx or 5xx)

        data = response.json()
        logger.debug(f"Received quote data: {data}")

        # Build a dictionary mapping symbols to their quote data
        quote_info = {}
        for item in data:
            symbol = item.get('symbol')
            if symbol:
                # Extract the required fields
                quote_info[symbol] = {
                    'price': item.get('price'),
                    'changesPercentage': item.get('changesPercentage'),
                    'marketCap': item.get('marketCap'),
                    'volume': item.get('volume'),
                    'exchange': item.get('exchange')
                }
        logger.info(f"Constructed quote_info dictionary with {len(quote_info)} entries.")
        return quote_info

    except Exception as e:
        logger.error(f"Error fetching quote info from FMP API: {e}")
        return {}