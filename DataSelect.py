import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from retrying import retry
from datetime import datetime

@retry(stop_max_attempt_number=5, wait_fixed=2000)
def fetch_uniprot_data(query, columns, size=500, offset=0):
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "format": "tsv",
        "fields": columns,
        "size": size,
        "offset": offset
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.text

def parse_tsv_data(tsv_data):
    from io import StringIO
    return pd.read_csv(StringIO(tsv_data), delimiter='\t')

def fetch_and_parse(query, columns, size, offset, index, total):
    tsv_data = fetch_uniprot_data(query, columns, size, offset)
    sys.stdout.write(f"\rCompleted request {index + 1}/{total}")
    sys.stdout.flush()
    return parse_tsv_data(tsv_data)

def fetch_uniprot_entries(query, columns, size=500, num_workers=8, output_file="output.csv"):
    initial_response = requests.get(f"https://rest.uniprot.org/uniprotkb/search?query={query}&format=tsv&size=1&fields=accession")
    total_entries = int(initial_response.headers['x-total-results'])
    print(f"Total entries: {total_entries}")

    offsets = list(range(0, total_entries, size))

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(fetch_and_parse, query, columns, size, offset, i, len(offsets)) for i, offset in enumerate(offsets)]
        for future in as_completed(futures):
            results.append(future.result())

    df = pd.concat(results, ignore_index=True)
    df.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")

start_time = datetime.now()
print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

fetch_uniprot_entries(
    query="reviewed:true",
    columns="accession,sequence,ec",
    size=500,
    num_workers=8,
    output_file="uniprot_swiss_enzymes.csv"
)

end_time = datetime.now()
print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")