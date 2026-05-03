import sys

try:
    from langchain.chains import RetrievalQA
    print("1. Found in langchain.chains")
except Exception as e:
    print(f"1. Not found: {e}")

try:
    from langchain_community.chains import RetrievalQA
    print("2. Found in langchain_community.chains")
except Exception as e:
    print(f"2. Not found: {e}")

try:
    from langchain.chains.retrieval_qa.base import RetrievalQA
    print("3. Found in langchain.chains.retrieval_qa.base")
except Exception as e:
    print(f"3. Not found: {e}")

try:
    from langchain_core.runnables import RunnablePassthrough
    print("4. Can use RunnablePassthrough (new API)")
except Exception as e:
    print(f"4. Not available: {e}")

# List available chains
try:
    import langchain.chains as chains_mod
    print("\nAvailable in langchain.chains:")
    items = [item for item in dir(chains_mod) if not item.startswith('_')]
    for item in items[:20]:
        print(f"  - {item}")
    if len(items) > 20:
        print(f"  ... and {len(items) - 20} more")
except Exception as e:
    print(f"Error listing: {e}")
