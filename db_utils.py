import os
import logging
import snowflake.connector
from snowflake.connector import DictCursor
from pinecone import Pinecone

logger = logging.getLogger()

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

def get_source_info_from_snowflake(source_ids):
    """
    Fetches site and url information for the given source_ids from Snowflake.
    """
    # Retrieve Snowflake credentials from environment variables
    SNOWFLAKE_ACCOUNT = os.environ.get('SNOWFLAKE_ACCOUNT')
    SNOWFLAKE_USER = os.environ.get('SNOWFLAKE_USER')
    SNOWFLAKE_PASSWORD = os.environ.get('SNOWFLAKE_PASSWORD')
    SNOWFLAKE_WAREHOUSE = os.environ.get('SNOWFLAKE_WAREHOUSE')
    SNOWFLAKE_DATABASE = os.environ.get('SNOWFLAKE_DATABASE')
    SNOWFLAKE_SCHEMA = os.environ.get('SNOWFLAKE_SCHEMA')
    SNOWFLAKE_ROLE = os.environ.get('SNOWFLAKE_ROLE')  # Optional
    
    # Validate credentials
    missing_vars = []
    required_vars = {
        'SNOWFLAKE_ACCOUNT': SNOWFLAKE_ACCOUNT,
        'SNOWFLAKE_USER': SNOWFLAKE_USER,
        'SNOWFLAKE_PASSWORD': SNOWFLAKE_PASSWORD,
        'SNOWFLAKE_WAREHOUSE': SNOWFLAKE_WAREHOUSE,
        'SNOWFLAKE_DATABASE': SNOWFLAKE_DATABASE,
        'SNOWFLAKE_SCHEMA': SNOWFLAKE_SCHEMA
    }
    
    for var_name, var_value in required_vars.items():
        if not var_value:
            missing_vars.append(var_name)
    
    if missing_vars:
        logger.error(f"Snowflake credentials are missing in environment variables: {', '.join(missing_vars)}")
        raise ValueError("Snowflake credentials are missing.")
    
    try:
        # Establish Snowflake connection
        ctx = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA,
            role=SNOWFLAKE_ROLE,  # This can be None if not set
            client_session_keep_alive=True  # Optional: Keeps the session alive
        )
        logger.info("Connected to Snowflake successfully.")

        logger.info(f'Searching for source_ids: {source_ids}')
        # Prepare the query
        source_ids_str = ",".join(f"'{sid}'" for sid in source_ids)
        query = f"""
        SELECT SOURCE_ID, SITE, URL
        FROM STOCK_NEWS
        WHERE SOURCE_ID IN ({source_ids_str})
        """
        logger.info("Executing query to fetch site and url for source_ids.")

        # Execute the query
        cursor = ctx.cursor(DictCursor)
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        ctx.close()

        logger.info(f"Fetched {len(results)} rows from Snowflake.")

        # Build the dictionary
        source_info = {}
        for row in results:
            source_id = row['SOURCE_ID']
            site = row['SITE']
            url = row['URL']
            source_info[source_id] = {'site': site, 'url': url}

        return source_info

    except Exception as e:
        logger.error(f"Error fetching source info from Snowflake: {str(e)}")
        raise
