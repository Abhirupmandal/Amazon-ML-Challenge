"""
Bounded candidate retrieval module using DuckDB.
Phase 3: Incremental candidate retrieval via persistent blocking index lookups.
Phase 4: Candidate compression, deterministic ranking, and bounded top-K pruning.
Uses 8 matching blocking rules for high recall.
"""

from pathlib import Path
from typing import List, Tuple
import duckdb

try:
    from src.config import CANDIDATE_TOP_K
except ImportError:
    from .config import CANDIDATE_TOP_K


def ensure_s1_table(con: duckdb.DuckDBPyConnection, s1_path: Path, table_name: str) -> int:
    """
    Ensure a sequentially ordered S1 table with row_idx exists in DuckDB.
    Guarantees deterministic, sequential slicing without duplicates or missing rows.
    Extracts the same normalized fields and name tokens as target_entities.
    """
    exists = con.execute(
        f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
    ).fetchone()[0]

    if exists > 0:
        total = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
        return total

    print(f"  Creating ordered S1 table '{table_name}' from {s1_path.name}...")
    s1_posix = s1_path.resolve().as_posix()
    con.execute(f"""
        CREATE TABLE {table_name} AS
        SELECT 
            row_number() OVER () AS row_idx,
            entity_id AS s1_id,
            business_name AS s1_name,
            business_address AS s1_addr,
            country AS s1_country,
            upper(trim(coalesce(country, ''))) AS country_norm,
            regexp_replace(lower(coalesce(business_name, '')), '[^a-z0-9]', '', 'g') AS name_compact,
            regexp_replace(lower(CASE 
                WHEN regexp_matches(coalesce(business_name, ''), '[^ -~]') THEN py_unidecode(business_name)
                ELSE coalesce(business_name, '')
            END), '[^a-z0-9]', '', 'g') AS latin_compact,
            regexp_replace(lower(coalesce(business_address, '')), '[^a-z0-9]', '', 'g') AS addr_compact,
            regexp_extract(coalesce(business_address, ''), '[0-9]+') AS addr_num,
            regexp_extract(lower(coalesce(business_address, '')), '[a-z]{{3,}}') AS addr_word,
            -- First significant name token (>= 4 chars, non-numeric)
            regexp_extract(
                regexp_replace(lower(CASE 
                    WHEN regexp_matches(coalesce(business_name, ''), '[^ -~]') THEN py_unidecode(business_name)
                    ELSE coalesce(business_name, '')
                END), '[^a-z ]', '', 'g'),
                '\\b([a-z]{{4,}})\\b'
            ) AS name_tok1,
            -- Second significant name token
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
        FROM read_csv('{s1_posix}', delim='\t', header=true, 
                       columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}, 
                       quote='', escape='', parallel=false);
    """)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_row ON {table_name}(row_idx);")
    total = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    print(f"  Ordered S1 table ready: {total:,} rows.")
    return total


def retrieve_candidates_batch(
    con: duckdb.DuckDBPyConnection,
    s1_table: str,
    start_row: int,
    end_row: int,
    top_k: int = CANDIDATE_TOP_K
) -> Tuple[List[str], List[Tuple]]:
    """
    Retrieve bounded top-K candidates for a deterministic range of S1 entities.
    Uses 8 blocking rules matching the indexing module for high recall.
    Returns:
      s1_ids: Ordered list of all S1 entity IDs in this slice (to ensure every S1 is output)
      candidate_rows: List of tuples with candidate attributes for feature extraction:
        (s1_id, cand_id, matched_rules, has_exact_rule,
         s1_name, s1_nc, s1_lc, s1_addr, s1_ac, s1_c,
         c_name, c_nc, c_lc, c_addr, c_ac, c_c)
    """
    # 1. Slice S1 chunk deterministically by row_idx
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_chunk AS
        SELECT * FROM {s1_table}
        WHERE row_idx BETWEEN {start_row} AND {end_row};
    """)

    # Get ordered list of all S1 IDs in this chunk
    s1_ids = [r[0] for r in con.execute("SELECT s1_id FROM s1_chunk ORDER BY row_idx").fetchall()]

    # 2. Extract multi-rule blocking keys for this S1 chunk (8 rules matching target index)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s1_keys AS
        -- Rule 1: Latin compact name + country
        SELECT s1_id, latin_compact || '|' || country_norm AS bkey, 1 AS rule_id
        FROM s1_chunk WHERE length(latin_compact) >= 3 AND country_norm != ''
        
        UNION ALL
        
        -- Rule 2: Native compact name + country (when different)
        SELECT s1_id, name_compact || '|' || country_norm AS bkey, 2 AS rule_id
        FROM s1_chunk WHERE length(name_compact) >= 3 AND name_compact != latin_compact AND country_norm != ''
        
        UNION ALL
        
        -- Rule 3: Prefix-6 of latin name + country
        SELECT s1_id, left(latin_compact, 6) || '|' || country_norm AS bkey, 3 AS rule_id
        FROM s1_chunk WHERE length(latin_compact) >= 6 AND country_norm != ''
        
        UNION ALL
        
        -- Rule 4: Address number + address word + country
        SELECT s1_id, addr_num || '|' || addr_word || '|' || country_norm AS bkey, 4 AS rule_id
        FROM s1_chunk WHERE addr_num IS NOT NULL AND addr_word IS NOT NULL 
          AND length(addr_num) > 0 AND length(addr_word) >= 3 AND country_norm != ''
        
        UNION ALL
        
        -- Rule 5: First significant name token + country
        SELECT s1_id, name_tok1 || '|' || country_norm AS bkey, 5 AS rule_id
        FROM s1_chunk WHERE name_tok1 IS NOT NULL AND length(name_tok1) >= 4 AND country_norm != ''
        
        UNION ALL
        
        -- Rule 6: Second significant name token + country
        SELECT s1_id, name_tok2 || '|' || country_norm AS bkey, 6 AS rule_id
        FROM s1_chunk WHERE name_tok2 IS NOT NULL AND length(name_tok2) >= 4 AND country_norm != ''
        
        UNION ALL
        
        -- Rule 7: Address prefix-10 + country
        SELECT s1_id, left(addr_compact, 10) || '|' || country_norm AS bkey, 7 AS rule_id
        FROM s1_chunk WHERE length(addr_compact) >= 10 AND country_norm != ''
        
        UNION ALL
        
        -- Rule 8: Sorted first-3 chars of latin name + country (n-gram proxy)
        SELECT s1_id, list_sort(list_value(left(latin_compact, 1), substr(latin_compact, 2, 1), substr(latin_compact, 3, 1)))::VARCHAR || '|' || country_norm AS bkey, 8 AS rule_id
        FROM s1_chunk WHERE length(latin_compact) >= 3 AND country_norm != '';
    """)

    # 3. Join with persistent target_postings index and rank top-K
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_candidates AS
        WITH hits AS (
            SELECT 
                s.s1_id,
                p.entity_id AS cand_id,
                count(distinct p.rule_id) AS matched_rules,
                max(CASE WHEN p.rule_id IN (1, 2) THEN 1 ELSE 0 END) AS has_exact_rule
            FROM s1_keys s
            JOIN target_postings p ON s.bkey = p.bkey
            GROUP BY s.s1_id, p.entity_id
        ),
        ranked AS (
            SELECT 
                s1_id,
                cand_id,
                matched_rules,
                has_exact_rule,
                row_number() OVER (PARTITION BY s1_id ORDER BY has_exact_rule DESC, matched_rules DESC) AS rk
            FROM hits
        )
        SELECT s1_id, cand_id, matched_rules, has_exact_rule
        FROM ranked
        WHERE rk <= {top_k};
    """)

    # 4. Join candidates with target_entities and s1_chunk to fetch attributes for ML features
    candidate_rows = con.execute("""
        SELECT 
            c.s1_id,
            c.cand_id,
            c.matched_rules,
            c.has_exact_rule,
            s.s1_name,
            s.name_compact AS s1_name_compact,
            s.latin_compact AS s1_latin_compact,
            s.s1_addr,
            s.addr_compact AS s1_addr_compact,
            s.country_norm AS s1_country,
            t.business_name AS cand_name,
            t.name_compact AS cand_name_compact,
            t.latin_compact AS cand_latin_compact,
            t.business_address AS cand_addr,
            t.addr_compact AS cand_addr_compact,
            t.country_norm AS cand_country
        FROM s1_candidates c
        JOIN s1_chunk s ON c.s1_id = s.s1_id
        JOIN target_entities t ON c.cand_id = t.entity_id;
    """).fetchall()

    return s1_ids, candidate_rows
