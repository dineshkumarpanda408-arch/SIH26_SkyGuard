"""Station metadata shared across the app (name, lat, lon)."""

STATION_META = [
    ("AWS-001", "Puri", 19.81, 85.83),
    ("AWS-002", "Khurda", 20.18, 85.77),
    ("AWS-003", "Cuttack", 20.46, 85.88),
    ("AWS-004", "Jharsuguda", 21.85, 84.03),
    ("AWS-005", "Rourkela", 22.26, 84.85),
    ("AWS-006", "Sambalpur", 21.47, 83.97),
    ("AWS-007", "Balasore", 21.49, 86.93),
    ("AWS-008", "Bhadrak", 21.05, 86.72),
    ("AWS-009", "Baripada", 21.93, 86.75),
    ("AWS-010", "Koraput", 18.81, 82.71),
    ("AWS-011", "Phulbani", 20.47, 84.24),
    ("AWS-012", "Keonjhar", 21.63, 85.59),
    ("AWS-013", "Angul", 20.84, 85.10),
    ("AWS-014", "Dhenkanal", 20.73, 85.69),
    ("AWS-015", "Nayagarh", 20.13, 85.11),
    ("AWS-016", "Ganjam", 19.38, 85.05),
    ("AWS-017", "Gajapati", 18.89, 84.18),
    ("AWS-018", "Rayagada", 19.17, 83.42),
    ("AWS-019", "Nuapada", 20.82, 82.54),
    ("AWS-020", "Bolangir", 20.71, 83.49),
    ("AWS-021", "Subarnapur", 20.86, 83.92),
    ("AWS-022", "Bargarh", 21.34, 83.62),
    ("AWS-023", "Bhubaneswar", 20.30, 85.82),
    ("AWS-024", "Jajpur", 20.85, 86.34),
]

META_BY_ID = {sid: {"name": n, "latitude": lat, "longitude": lon} for sid, n, lat, lon in STATION_META}

# Approximate station elevation in metres (used by the synthetic generator to
# derive a physically plausible pressure baseline). Approximate, roughly
# representative of the Odisha distribution; NOT survey-grade.
STATION_ELEV = {
    "AWS-001": 8, "AWS-002": 30, "AWS-003": 25, "AWS-004": 200,
    "AWS-005": 219, "AWS-006": 143, "AWS-007": 16, "AWS-008": 23,
    "AWS-009": 40, "AWS-010": 870, "AWS-011": 485, "AWS-012": 470,
    "AWS-013": 120, "AWS-014": 75, "AWS-015": 89, "AWS-016": 3,
    "AWS-017": 60, "AWS-018": 250, "AWS-019": 250, "AWS-020": 204,
    "AWS-021": 160, "AWS-022": 183, "AWS-023": 45, "AWS-024": 25,
}
