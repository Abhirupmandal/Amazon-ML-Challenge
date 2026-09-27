"""
Persistent disk-backed blocking index module using DuckDB.
Phase 2: Builds reusable, disk-backed indexes over Source 2 and Source 3
with frequency statistics and frequency pruning to avoid false candidate explosion.
"""

import os
import time
from pathlib import Path
from typing import Optional

import duckdb
from unidecode import unidecode

try:
    from src.config import (
        DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS, MAX_KEY_FREQUENCY
    )
except ImportError:
    from .config import (
        DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS, MAX_KEY_FREQUENCY
    )


def safe_unidecode(s: Optional[str]) -> str:
    """Safe Unicode transliteration for non-ASCII business names."""
    if not s:
        return ""
    try:
        return unidecode(s)
    except Exception:
        return s


def is_index_built(db_path: Path) -> bool:
    """Check if the persistent DuckDB index exists and is populated."""
    if not db_path.exists() or db_path.stat().st_size == 0:
        return False
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        res = con.execute("SELECT count(*) FROM target_postings").fetchone()
        con.close()
        return res is not None and res[0] > 0
    except Exception:
        return False


def get_duckdb_connection(db_path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open an optimized DuckDB connection with configured memory and thread limits."""
    con = duckdb.connect(str(db_path), read_only=read_only)
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order=false")
    con.create_function("py_unidecode", safe_unidecode, ["VARCHAR"], "VARCHAR")
    return con


def build_persistent_index(
    db_path: Path,
    source2_path: Path,
    source3_path: Path,
    max_key_frequency: int = MAX_KEY_FREQUENCY,
    force_rebuild: bool = False
) -> int:
    """
    Build a persistent disk-backed DuckDB index for Source 2 and Source 3.
    Stores normalized entity attributes and frequency-pruned blocking postings.
    Uses 8 blocking rules for high recall:
      R1: Latin compact name + country (exact match)
      R2: Native compact name + country (when different from latin)
      R3: Prefix-6 of latin name + country (fuzzy prefix)
      R4: Address number + address word + country
      R5: First significant name token (>=4 chars) + country
      R6: Second significant name token (>=4 chars) + country
      R7: Address prefix-10 + country
      R8: Sorted first-3 chars of latin name + country (character n-gram proxy)
    Returns total entities indexed.
    """
    if not force_rebuild and is_index_built(db_path):
        con = duckdb.connect(str(db_path), read_only=True)
        total = con.execute("SELECT count(*) FROM target_entities").fetchone()[0]
        postings = con.execute("SELECT count(*) FROM target_postings").fetchone()[0]
        con.close()
        print(f"  Reusing existing DuckDB index: {db_path.name} ({total:,} entities, {postings:,} postings)")
        return total

    if db_path.exists():
        try:
            os.remove(db_path)
        except OSError:
            pass

    print(f"\nBuilding persistent DuckDB index: {db_path.name}")
    start = time.perf_counter()

    con = get_duckdb_connection(db_path, read_only=False)

    # 1. Ingest and normalize S2 + S3 entities into columnar storage
    print("  Step 1: Normalizing and indexing target entities (S2 + S3)...")
    s2_posix = source2_path.resolve().as_posix()
    s3_posix = source3_path.resolve().as_posix()

    con.execute(f"""
        CREATE TABLE target_entities AS
        SELECT 
            entity_id,
            business_name,
            business_address,
            country,
            upper(trim(coalesce(country, ''))) AS country_norm,
            regexp_replace(lower(coalesce(business_name, '')), '[^a-z0-9]', '', 'g') AS name_compact,
            regexp_replace(lower(CASE 
                WHEN regexp_matches(coalesce(business_name, ''), '[^ -~]') THEN py_unidecode(business_name)
                ELSE coalesce(business_name, '')
            END), '[^a-z0-9]', '', 'g') AS latin_compact,
            regexp_replace(lower(coalesce(business_address, '')), '[^a-z0-9]', '', 'g') AS addr_compact,
            regexp_extract(coalesce(business_address, ''), '[0-9]+') AS addr_num,
            regexp_extract(lower(coalesce(business_address, '')), '[a-z]{{3,}}') AS addr_word,
            -- Extract first significant token (>= 4 chars, non-numeric) from latin-lowered name
            regexp_extract(
                regexp_replace(lower(CASE 
                    WHEN regexp_matches(coalesce(business_name, ''), '[^ -~]') THEN py_unidecode(business_name)
                    ELSE coalesce(business_name, '')
                END), '[^a-z ]', '', 'g'),
                '\\b([a-z]{{4,}})\\b'
            ) AS name_tok1,
            -- Extract second significant token via regex on remaining text after first token
            regexp_extract(
                regexp_replace(
                    regexp_replace(lower(CASE 
                        WHEN regexp_matches(coalesce(business_name, ''), '[^ -~]') THEN py_unidecode(business_name)
                        ELSE coalesce(business_name, '')
                    END), '[^a-z ]', '', 'g'),
                    '\\b[a-z]{{4,}}\\b', '', 'g'
                ),
                '\\b([a-z]{{4,}})\\b'
            ) AS name_tok2
        FROM (
            SELECT * FROM read_csv('{s2_posix}', delim='\t', header=true, 
                                   columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}, 
                                   quote='', escape='', parallel=true)
            UNION ALL
            SELECT * FROM read_csv('{s3_posix}', delim='\t', header=true, 
                                   columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}, 
                                   quote='', escape='', parallel=true)
        ) t;
    """)

    entity_count = con.execute("SELECT count(*) FROM target_entities").fetchone()[0]
    e_time = time.perf_counter() - start
    print(f"  Step 1 complete: {entity_count:,} entities indexed in {e_time:.1f}s ({entity_count/max(e_time,0.1):,.0f} rows/s)")

    # 2. Extract multi-rule blocking keys with frequency pruning (8 rules)
    print(f"  Step 2: Building frequency-pruned blocking postings (max_freq <= {max_key_frequency}, 8 rules)...")
    p_start = time.perf_counter()

    con.execute(f"""
        CREATE TABLE target_postings AS
        WITH raw_p AS (
            -- Rule 1: Latin compact name + country (exact match)
            SELECT latin_compact || '|' || country_norm AS bkey, entity_id, 1 AS rule_id
            FROM target_entities 
            WHERE length(latin_compact) >= 3 AND country_norm != ''
            
            UNION ALL
            
            -- Rule 2: Native compact name + country (if distinct from latin)
            SELECT name_compact || '|' || country_norm AS bkey, entity_id, 2 AS rule_id
            FROM target_entities 
            WHERE length(name_compact) >= 3 AND name_compact != latin_compact AND country_norm != ''
            
            UNION ALL
            
            -- Rule 3: Prefix 6 chars of latin name + country
            SELECT left(latin_compact, 6) || '|' || country_norm AS bkey, entity_id, 3 AS rule_id
            FROM target_entities 
            WHERE length(latin_compact) >= 6 AND country_norm != ''
            
            UNION ALL
            
            -- Rule 4: Address numeric token + address word + country
            SELECT addr_num || '|' || addr_word || '|' || country_norm AS bkey, entity_id, 4 AS rule_id
            FROM target_entities 
            WHERE addr_num IS NOT NULL AND addr_word IS NOT NULL 
              AND length(addr_num) > 0 AND length(addr_word) >= 3 AND country_norm != ''
            
            UNION ALL
            
            -- Rule 5: First significant name token (>= 4 chars) + country
            SELECT name_tok1 || '|' || country_norm AS bkey, entity_id, 5 AS rule_id
            FROM target_entities
            WHERE name_tok1 IS NOT NULL AND length(name_tok1) >= 4 AND country_norm != ''
            
            UNION ALL
            
            -- Rule 6: Second significant name token (>= 4 chars) + country
            SELECT name_tok2 || '|' || country_norm AS bkey, entity_id, 6 AS rule_id
            FROM target_entities
            WHERE name_tok2 IS NOT NULL AND length(name_tok2) >= 4 AND country_norm != ''
            
            UNION ALL
            
            -- Rule 7: Address prefix-10 + country
            SELECT left(addr_compact, 10) || '|' || country_norm AS bkey, entity_id, 7 AS rule_id
            FROM target_entities
            WHERE length(addr_compact) >= 10 AND country_norm != ''
            
            UNION ALL
            
            -- Rule 8: Sorted first-3 chars of latin name + country (n-gram proxy for fuzzy)
            SELECT list_sort(list_value(left(latin_compact, 1), substr(latin_compact, 2, 1), substr(latin_compact, 3, 1)))::VARCHAR || '|' || country_norm AS bkey, entity_id, 8 AS rule_id
            FROM target_entities
            WHERE length(latin_compact) >= 3 AND country_norm != ''
        ),
        counts AS (
            SELECT bkey, count(*) AS cnt 
            FROM raw_p 
            GROUP BY bkey
        )
        SELECT r.bkey, r.entity_id, r.rule_id
        FROM raw_p r
        JOIN counts c ON r.bkey = c.bkey
        WHERE c.cnt <= {max_key_frequency};
    """)

    postings_count = con.execute("SELECT count(*) FROM target_postings").fetchone()[0]
    p_time = time.perf_counter() - p_start
    print(f"  Step 2 complete: {postings_count:,} postings built in {p_time:.1f}s")

    # 3. Create index for fast lookups
    print("  Step 3: Creating index on blocking keys...")
    idx_start = time.perf_counter()
    con.execute("CREATE INDEX idx_postings_bkey ON target_postings(bkey);")
    idx_time = time.perf_counter() - idx_start
    print(f"  Step 3 complete: Index created in {idx_time:.1f}s")

    con.close()

    total_time = time.perf_counter() - start
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"Persistent DuckDB index ready: {db_path.name} ({db_size_mb:.1f} MB) in {total_time:.1f}s ({total_time/60:.2f} min)\n")

    return entity_count
