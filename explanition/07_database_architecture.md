# 07. Database Architecture

The Face Recognition System does **not** use a traditional database management system (like SQL, MongoDB, etc.). Instead, it uses a **file-based storage architecture** relying on local CSV files and image saving.

## Architecture

```text
callback()
 ↓
File System Directory Check (`marked_attendance/YYYY_MM_DD`)
 ↓
CSV File Check (`YYYY_MM_DD_attendance_sheet.csv`)
 ↓
Save Cropped Face (.jpg)
 ↓
Append Row to CSV
```

## Schema (CSV)

Each day, a new directory is created (`marked_attendance/YYYY_MM_DD`). Inside, a CSV file named `YYYY_MM_DD_attendance_sheet.csv` is maintained.

### Fields:
1. **Name**: `string` - The recognized name of the employee (e.g., `JohnDoe`).
2. **UniqueID**: `string (UUID)` - A randomly generated UUID for the transaction (e.g., `123e4567-e89b-12d3-a456-426614174000`).
3. **Timestamp**: `string` - Exact date and time (`YYYY_MM_DD_HH:MM:SS`).
4. **Hyperlink**: `string` - The absolute file path to the saved cropped face image (`/path/to/marked_attendance/YYYY_MM_DD/Name_Timestamp.jpg`).

## Storage Operations

### Create/Initialize:
When a face crosses the line, the system checks if the directory and CSV exist:
```python
output_dir = os.path.join('marked_attendance', current_date)
os.makedirs(output_dir, exist_ok=True)
```
If the CSV doesn't exist, it writes the headers: `['Name', 'UniqueID', 'Timestamp', 'Hyperlink']`.

### Insert (Append):
If the person is recognized with `>= 0.87` probability and hasn't been logged in the `saved_names` memory array during the current run:
```python
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([first_name[0], unique_id, timestamp, hyperlink])
```

## Image Storage
Cropped faces are saved directly to the folder:
`marked_attendance/YYYY_MM_DD/<Name>_<Timestamp>.jpg`

## Limitations
- State (`saved_names`) is kept in memory. Restarting the script clears the memory, allowing duplicate attendance logs for the same day.
- No database indexes, querying, or built-in validation.
