CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    upload_date TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    file_size_kb REAL,
    mean_r REAL, mean_g REAL, mean_b REAL,
    contrast REAL,
    edge_density REAL,
    hist TEXT,                            -- JSON: 16-bin gray/r/g/b histograms
    manual_label TEXT,                    -- 'clean' | 'dirty' | NULL
    auto_label TEXT,                      -- 'clean' | 'dirty' | NULL  (rule-based)
    confidence REAL,                      -- NULL until classified
    rule_reason TEXT,                     -- why the rule engine decided
    latitude REAL, longitude REAL
);

-- Key/value store for configurable rule thresholds (edited from /rules).
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
