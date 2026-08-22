
import os , json
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.identity import DefaultAzureCredential, ClientSecretCredential

load_dotenv() 

account_url    = os.getenv("AZURE_STORAGE_URL")
container_name = os.getenv("AZURE_CONTAINER_NAME")
blob_name      = os.getenv("AZURE_BLOB_NAME")

client_id = os.getenv("AZURE_CLIENT_ID")
tenant_id = os.getenv("AZURE_TENANT_ID")
client_secret = os.getenv("AZURE_CLIENT_SECRET")
vault_url = os.getenv("AZURE_VAULT_URL")
storage_key    = os.getenv("AZURE_STORAGE_KEY") 


credentials = ClientSecretCredential(
                client_id     = client_id,
                client_secret = client_secret,
                tenant_id     = tenant_id

                                        )

def upload_blob_data(payload, company, year):

    # set client to access azure storage container
    blob_service_client = BlobServiceClient(account_url = account_url, credential = storage_key)

    # Then using blob service client get container client and access that container
    container_client = blob_service_client.get_container_client(container = container_name)

    # download blob data using container client and access blob or file
   
    dynamic_blob_path = f"{blob_name}/{company}_{year}_extracted_kpis.json"
    blob_client = container_client.get_blob_client(blob = dynamic_blob_path)

    # read all information from this blob then
    payload = payload.encode('utf-8')
    data = blob_client.upload_blob(payload, overwrite = True, 
                                   content_settings = ContentSettings(content_type = "application/json") )

    return data

#secret_name = "secret"
#secret_client = SecretClient(vault_url = vault_url, credential = credentials)
#secret = secret_client.get_secret(secret_name)


def get_blob_data() -> list[dict]:
    """List every extracted-KPI blob under the `{blob_name}/` prefix and
    download+parse each one, without needing to know company/year ahead
    of time.
    """
    blob_service_client = BlobServiceClient(account_url=account_url, credential=storage_key)
    container_client = blob_service_client.get_container_client(container=container_name)
 
    all_data = []
 
    # name_starts_with acts as a folder-prefix filter server-side, so this
    # only lists blobs under e.g. "metrics/", not the whole container.
    blob_list = container_client.list_blobs(name_starts_with=f"{blob_name}/")
 
    for blob_props in blob_list:
        blob_client = container_client.get_blob_client(blob=blob_props.name)
        try:
            raw = blob_client.download_blob().readall().decode("utf-8")
            parsed = json.loads(raw)
        except Exception as e:
            print(f"[WARN] failed to read/parse blob '{blob_props.name}': {e}")
            continue
 
        # Pull company/year back out of the filename, since the JSON
        # payload itself may not carry them.
        # expected format: "<company>_<year>_extracted_kpis.json"
        filename = blob_props.name.split("/")[-1]
        stem = filename.removesuffix("_extracted_kpis.json")
        company, _, year = stem.rpartition("_")
 
        all_data.append({
            "company": company or None,
            "year": year or None,
            "blob_name": blob_props.name,
            "metrics": parsed,
        })
 
    print(f"[INFO] loaded {len(all_data)} document(s) from container '{container_name}'")
    return all_data

# def get_blob_data(company,year):

#     # set client to access azure storage container
#     blob_service_client = BlobServiceClient(account_url = account_url, credential = storage_key)

#     # Then using blob service client get container client and access that container
#     container_client = blob_service_client.get_container_client(container = container_name)

#     # download blob data using container client and access blob or file
#     dynamic_blob_path = f"{blob_name}/{company}_{year}_extracted_kpis.json"
#     blob_client = container_client.get_blob_client(blob = dynamic_blob_path)

#     # read all information from this blob then
#     data = blob_client.download_blob().readall().decode("utf-8")

#     return data

