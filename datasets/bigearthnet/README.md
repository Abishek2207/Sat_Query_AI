# BigEarthNet VQA Dataset

## Dataset Purpose
This dataset provides the foundational multimodal metadata and remote-sensing question-answer pairs required for the SIH 2026 Problem Statement 26167 (SatQuery AI) remote-sensing adaptation phase.

## Actual Schema
*   `ID` (int64)
*   `s1_name` (string): Sentinel-1 scene/patch identifier.
*   `patch_id` (string): Sentinel-2 patch identifier.
*   `input` (string): Natural-language remote-sensing question.
*   `output` (string): The corresponding answer.
*   `type` (string): Question type (e.g., binary).
*   `category` (string): Question category (e.g., adjacency, area).
*   `split` (string): Dataset split (train, val, test).
*   `latitude`, `longitude` (double): Geographic coordinates.
*   `country`, `season`, `climate_zone` (string): Contextual metadata.

## Crucial Semantics
**This Parquet file contains metadata and question-answer records, NOT raw satellite image arrays.**
The fields `s1_name` and `patch_id` are strictly identifiers. During actual model training, an external image loader must use these identifiers to fetch the physical GeoTIFF/PNG files from a separate storage location.

## Loader Architecture
The `BigEarthNetLoader` uses `pyarrow.parquet.ParquetFile` to stream row groups lazily. It never loads the entire 467 MB dataset into RAM simultaneously, ensuring the application remains lightweight on the Snapdragon X host machine.
