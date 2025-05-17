import os
import uuid
import shutil
from config import CACHE_DIR, MAX_CACHE_SIZE_MB

def ensure_dir_exists(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def generate_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())

def get_cache_size_mb(cache_dir=CACHE_DIR):
    """Get the current size of the cache directory in MB."""
    total_size = 0
    if os.path.exists(cache_dir):
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert bytes to MB

def clean_old_cache_files(cache_dir=CACHE_DIR, max_size_mb=MAX_CACHE_SIZE_MB):
    """Clean up old cache files if the cache directory exceeds the maximum size."""
    if not os.path.exists(cache_dir):
        return
    
    current_size_mb = get_cache_size_mb(cache_dir)
    
    if current_size_mb > max_size_mb:
        print(f"Cache size ({current_size_mb:.2f} MB) exceeds maximum ({max_size_mb} MB). Cleaning up...")
        
        # Get all files with creation time
        files_with_time = []
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    creation_time = os.path.getctime(filepath)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    files_with_time.append((filepath, creation_time, size_mb))
        
        # Sort by creation time (oldest first)
        files_with_time.sort(key=lambda x: x[1])
        
        # Delete oldest files until we're under the limit
        for filepath, _, size_mb in files_with_time:
            if current_size_mb <= max_size_mb * 0.8:  # Keep deleting until we're at 80% of max
                break
            
            try:
                os.remove(filepath)
                print(f"Deleted cache file: {filepath} ({size_mb:.2f} MB)")
                current_size_mb -= size_mb
            except Exception as e:
                print(f"Error deleting {filepath}: {e}")

def safe_filename(filename):
    """Convert a string to a safe filename."""
    return ''.join(c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in filename)
