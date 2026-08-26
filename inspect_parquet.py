import pyarrow.parquet as pq

dataset_path = r"datasets\bigearthnet\BigEarthNet.txt.parquet"

try:
    pf = pq.ParquetFile(dataset_path)

    print("--- DATASET INSPECTION ---")
    print(f"Total Rows: {pf.metadata.num_rows}")
    print(f"Total Row Groups: {pf.metadata.num_row_groups}")

    print("\n--- SCHEMA ---")
    schema = pf.schema_arrow

    for field in schema:
        print(f"Column: {field.name} | Type: {field.type}")

    print("\n--- FIRST 3 SAMPLES ---")
    first_batch = next(pf.iter_batches(batch_size=3))
    df = first_batch.to_pandas()

    for index, row in df.iterrows():
        print(f"\nSample {index + 1}:")
        print(row.to_dict())

except Exception as e:
    print(f"Failed to inspect Parquet: {type(e).__name__}: {e}")
