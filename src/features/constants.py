from __future__ import annotations

FEATURE_COLUMNS = [
    "connections",
    "bytes",
    "packets",
    "packet_size",
    "ports",
]

RAW_TO_FEATURE_MAPPING = {
    "Flow Duration": "connections",
    "Total Length of Fwd Packets": "bytes",
    "Total Fwd Packets": "packets",
    "Average Packet Size": "packet_size",
    "Destination Port": "ports",
}

KNOWN_LABEL_COLUMNS = [
    "label",
    "Label",
    "final_anomaly",
    "anomaly",
    "is_anomaly",
]

BENIGN_TEXT_LABELS = {
    "BENIGN",
    "NORMAL",
    "NORMAL_TRAFFIC",
    "NORMAL TRAFFIC",
    "0",
}
