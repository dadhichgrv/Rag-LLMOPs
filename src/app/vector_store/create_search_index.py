import os   
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
                                                    HnswAlgorithmConfiguration,
                                                    SearchField,
                                                    SearchFieldDataType,
                                                    SearchIndex,
                                                    SimpleField,
                                                    VectorSearch,
                                                    VectorSearchProfile
                                                    )

load_dotenv()

index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
embedding_dimensions = 1024

def create_index():
   
    search_index_client = SearchIndexClient(   
                                 credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY")),
                                 endpoint   = os.getenv("AZURE_SEARCH_ENDPOINT")
                                 )
    
    fields = [
        SimpleField(name= "id"     ,        type = SearchFieldDataType.String, key = True),
        SimpleField(name= "company",        type = SearchFieldDataType.String, filterable = True),
        SimpleField(name= "year",           type = SearchFieldDataType.String, filterable = True),
        SimpleField(name= "source_file",    type = SearchFieldDataType.String, filterable = True),
        SearchField(name= "content",        type = SearchFieldDataType.String, searchable = True),
        SearchField(name= "content_vector", type = SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                    vector_search_dimensions = embedding_dimensions,
                    vector_search_profile_name = "vector-profile" 
                   )   
            ]
    
    vector_search = VectorSearch(
                    algorithms = [HnswAlgorithmConfiguration(name = "hnsw-config")],
                    profiles = [ VectorSearchProfile(
                                                name = "vector-profile",
                                                algorithm_configuration_name = "hnsw-config"
                              )]
                                )
    
    index = SearchIndex( name = index_name,
                        fields = fields,
                        vector_search = vector_search)
    
    search_index_client.create_or_update_index(index)

    print(f"Index {index_name} created successfully. ")


if __name__ == "__main__":
    create_index()


