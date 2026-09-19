# Proposed disk cleanup

Status: explicitly approved by the user and completed. All seven approved targets were verified absent. Approximately 9.67 GB was recovered; free space after cleanup was 15.15 GB. See cleanup-result.json for actual results.

Free space at this measurement: 5.49 GB.
Candidate bytes: 9.67 GB; projected free space: 15.15 GB.

Sizes are logical file sizes; actual recovered disk space may differ. Incomplete model downloads lose their resume progress when deleted. Completed model files, project datasets, the replacement ZIP, Docker data, application runtimes and personal documents are excluded.

## Approved targets (now deleted)

- `C:\Users\User\.cache\huggingface\hub\models--BAAI--bge-reranker-base\blobs\ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd.508469e8.incomplete`
  - incomplete model download; 0.268 GB; single file.
  - Last modified: 2026-08-10T15:03:20.311827.
- `C:\Users\User\.cache\huggingface\hub\models--BAAI--bge-reranker-base\blobs\ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd.98cf67f0.incomplete`
  - incomplete model download; 0.335 GB; single file.
  - Last modified: 2026-08-10T15:13:29.121533.
- `C:\Users\User\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\blobs\365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4.incomplete`
  - incomplete model download; 2.827 GB; single file.
  - Last modified: 2026-07-13T15:26:26.422958.
- `C:\Users\User\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\blobs\41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8.incomplete`
  - incomplete model download; 3.452 GB; single file.
  - Last modified: 2026-07-13T15:26:30.948489.
- `C:\Users\User\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3\blobs\69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1.incomplete`
  - incomplete model download; 2.328 GB; single file.
  - Last modified: 2026-08-07T13:57:32.772900.
- `C:\Users\User\.cache\huggingface\hub\models--Systran--faster-whisper-small\blobs\3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671.incomplete`
  - incomplete model download; 0.201 GB; single file.
  - Last modified: 2026-03-30T01:34:16.073318.
- `C:\Users\User\AppData\Local\pip\Cache\http-v2`
  - Python package download cache; 0.254 GB; directory contents.

Approval applies only to these paths. Before deletion, verify resolved targets remain within the listed Hugging Face model blob folders or pip cache root, reject reparse points, and check for resumed downloads. Use native PowerShell literal paths and report the actual resulting free space.
