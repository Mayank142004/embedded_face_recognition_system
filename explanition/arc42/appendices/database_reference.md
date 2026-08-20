# Appendix C: Database Reference

> [!NOTE]
> The system does not use a traditional SQL or NoSQL database. It relies entirely on the local file system using CSV files.

### Collection: `marked_attendance`

**Storage Path:** `marked_attendance/YYYY_MM_DD/YYYY_MM_DD_attendance_sheet.csv`

| Field | Data Type | Example | Description |
| ----- | --------- | ------- | ----------- |
| `Name` | `String` | `akshay` | The predicted name from the SVM classifier. |
| `UniqueID` | `UUID` (String) | `a1b2c3d4-e5f6-7890...` | A universally unique identifier generated at the moment of logging. |
| `Timestamp` | `String` | `2026_08_18_10:35:43` | The exact time the person crossed the attendance line. |
| `Hyperlink` | `String` | `/app/marked_attendance/2026_08_18/akshay_...jpg` | The absolute path to the saved `.jpg` image containing the cropped face proof. |

### Relationships

- **One-to-One:** Each row in the CSV corresponds to exactly one `.jpg` file saved in the same `YYYY_MM_DD` directory.
