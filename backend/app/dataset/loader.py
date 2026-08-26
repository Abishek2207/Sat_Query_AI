import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional

class BigEarthNetLoader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['dataset']
            
        self.file_path = Path(self.config['path'])
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
            
        self.pf = pq.ParquetFile(self.file_path)
        self._schema_names = self.pf.schema.names
        
        required_cols = ["ID", "s1_name", "patch_id", "input", "output", "split"]
        missing = [col for col in required_cols if col not in self._schema_names]
        if missing:
             raise ValueError(f"Parquet file is missing required real columns: {missing}")

    def get_schema(self) -> Dict[str, str]:
        arrow_schema = self.pf.schema.to_arrow_schema()
        return {name: str(type_) for name, type_ in zip(arrow_schema.names, arrow_schema.types)}

    def get_total_rows(self) -> int:
        return self.pf.metadata.num_rows

    def get_row_groups(self) -> int:
        return self.pf.num_row_groups

    def stream_split(self, split_name: str, columns: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        cols_to_load = columns if columns else self._schema_names
        if "split" not in cols_to_load:
             cols_to_load.append("split")
             
        for batch in self.pf.iter_batches(batch_size=self.config['batch_size'], columns=cols_to_load):
            df = batch.to_pandas()
            filtered_df = df[df['split'] == split_name]
            
            for _, row in filtered_df.iterrows():
                row_dict = row.to_dict()
                mapped_dict = {
                    "question": row_dict.get("input"),
                    "answer": row_dict.get("output"),
                    "sar_reference": row_dict.get("s1_name"),
                    "optical_patch_reference": row_dict.get("patch_id")
                }
                for k, v in row_dict.items():
                    if k not in ["input", "output", "s1_name", "patch_id"]:
                        mapped_dict[k] = v
                yield mapped_dict

    def sample_records(self, n: int, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        records = []
        np.random.seed(self.config['random_seed'])
        
        for batch in self.pf.iter_batches(batch_size=self.config['batch_size'], columns=columns):
            df = batch.to_pandas()
            sample_size = min(n - len(records), len(df))
            if sample_size > 0:
                 sampled_df = df.sample(n=sample_size, random_state=self.config['random_seed'])
                 for _, row in sampled_df.iterrows():
                     r_dict = row.to_dict()
                     r_dict["question"] = r_dict.get("input")
                     r_dict["answer"] = r_dict.get("output")
                     r_dict["sar_reference"] = r_dict.get("s1_name")
                     r_dict["optical_patch_reference"] = r_dict.get("patch_id")
                     records.append(r_dict)
            if len(records) >= n:
                 break
        return records
