import os
import sys
import requests
import time

def check_weaviate():
    """Check connection to Weaviate service"""
    weaviate_host = os.environ.get("WEAVIATE_HOST", "weaviate")
    weaviate_url = f"http://{weaviate_host}:8080/v1/.well-known/ready"
    
    try:
        response = requests.get(weaviate_url, timeout=5)
        if response.status_code == 200:
            print("✅ Weaviate connection successful")
            return True
        else:
            print(f"❌ Weaviate returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        return False

def check_ollama():
    """Check connection to Ollama service"""
    ollama_host = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
    if not ollama_host.startswith("http"):
        ollama_host = f"http://{ollama_host}"
    if not ollama_host.endswith(":11434"):
        ollama_host = f"{ollama_host}:11434"
    
    ollama_url = f"{ollama_host}/api/health"
    
    try:
        response = requests.get(ollama_url, timeout=5)
        if response.status_code == 200:
            print("✅ Ollama connection successful")
            return True
        else:
            print(f"❌ Ollama returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        return False

def main():
    """Check connections to required services"""
    print("Checking connections to services...")
    
    # Wait a bit for services to be available
    time.sleep(2)
    
    weaviate_ok = check_weaviate()
    ollama_ok = check_ollama()
    
    if weaviate_ok and ollama_ok:
        print("\n✅ All connections successful. Your services should work correctly.")
        return 0
    else:
        failed = []
        if not weaviate_ok:
            failed.append("Weaviate")
        if not ollama_ok:
            failed.append("Ollama")
            
        print(f"\n❌ Connection issues with: {', '.join(failed)}")
        print("Please check that these services are running.")
        return 1

if __name__ == "__main__":
    sys.exit(main())