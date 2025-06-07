# check_version.py

import inspect
import langchain_core
import langchain_postgres
from langchain_postgres.vectorstores import PGVector

print("--- Library Versions & Locations ---")
print(f"langchain-core version: {langchain_core.__version__}")
print(f"Location: {langchain_core.__file__}")
print("-" * 30)
print(f"langchain-postgres version: {langchain_postgres.__version__}")
print(f"Location: {langchain_postgres.__file__}")
print("-" * 30)

print("\n--- PGVector.__init__ Signature ---")
try:
    # พยายามพิมพ์ Signature ของฟังก์ชัน __init__ ของ PGVector
    signature = inspect.signature(PGVector.__init__)
    print("The function PGVector.__init__ requires the following parameters:")
    print(signature)
except Exception as e:
    print(f"Could not inspect PGVector.__init__. Error: {e}")