Files for demonstrating input validation:
- unsupported.docx / empty.txt -> rejected at intake (HTTP 422, shown in the UI)
- corrupt_photo.jpg -> accepted with a warning; image tool fails; limitation shown in the report
- scanned_no_text.pdf -> accepted; document tool reports no extractable text (no OCR)
