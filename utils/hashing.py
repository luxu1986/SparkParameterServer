"""
Hashing functions for embedding sharding across workers.
Implements consistent hashing to distribute embedding features.
"""

import hashlib
from typing import Union, List


def consistent_hash(feature_id: Union[int, str], num_workers: int) -> int:
    """
    Consistent hashing function to map feature IDs to worker IDs.
    
    This function ensures that:
    1. The same feature ID always maps to the same worker
    2. Features are distributed roughly evenly across workers
    3. The mapping is deterministic and reproducible
    
    Args:
        feature_id: The feature ID to hash (can be int or string)
        num_workers: Total number of workers
        
    Returns:
        Worker ID (0 to num_workers-1) that should handle this feature
        
    Raises:
        ValueError: If num_workers <= 0
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    
    # Convert feature_id to string for consistent hashing
    feature_str = str(feature_id)
    
    # Use SHA-256 for consistent hashing
    hash_object = hashlib.sha256(feature_str.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    
    # Convert first 8 characters of hex to integer
    hash_int = int(hash_hex[:8], 16)
    
    # Map to worker ID using modulo
    worker_id = hash_int % num_workers
    
    return worker_id


def hash_features_to_workers(feature_ids: List[Union[int, str]], 
                           num_workers: int) -> dict:
    """
    Hash multiple feature IDs to their corresponding workers.
    
    Args:
        feature_ids: List of feature IDs to hash
        num_workers: Total number of workers
        
    Returns:
        Dictionary mapping worker_id to list of feature_ids
    """
    worker_assignments = {}
    
    for feature_id in feature_ids:
        worker_id = consistent_hash(feature_id, num_workers)
        
        if worker_id not in worker_assignments:
            worker_assignments[worker_id] = []
        
        worker_assignments[worker_id].append(feature_id)
    
    return worker_assignments


def get_worker_for_feature(feature_id: Union[int, str], num_workers: int) -> int:
    """
    Get the worker ID responsible for a specific feature.
    This is an alias for consistent_hash for better readability.
    
    Args:
        feature_id: The feature ID
        num_workers: Total number of workers
        
    Returns:
        Worker ID that handles this feature
    """
    return consistent_hash(feature_id, num_workers)


def verify_hash_distribution(feature_ids: List[Union[int, str]], 
                           num_workers: int) -> dict:
    """
    Verify the distribution of features across workers.
    Useful for debugging and ensuring balanced load.
    
    Args:
        feature_ids: List of feature IDs to analyze
        num_workers: Total number of workers
        
    Returns:
        Dictionary with distribution statistics
    """
    worker_counts = {}
    worker_assignments = hash_features_to_workers(feature_ids, num_workers)
    
    # Count features per worker
    for worker_id in range(num_workers):
        worker_counts[worker_id] = len(worker_assignments.get(worker_id, []))
    
    total_features = len(feature_ids)
    expected_per_worker = total_features / num_workers
    
    # Calculate statistics
    min_count = min(worker_counts.values())
    max_count = max(worker_counts.values())
    variance = sum((count - expected_per_worker) ** 2 for count in worker_counts.values()) / num_workers
    
    return {
        'worker_counts': worker_counts,
        'total_features': total_features,
        'expected_per_worker': expected_per_worker,
        'min_count': min_count,
        'max_count': max_count,
        'variance': variance,
        'load_balance_ratio': min_count / max_count if max_count > 0 else 0.0
    }


# Alternative hashing strategies for different use cases

def simple_modulo_hash(feature_id: Union[int, str], num_workers: int) -> int:
    """
    Simple modulo-based hashing (less robust than consistent hashing).
    
    Args:
        feature_id: The feature ID to hash
        num_workers: Total number of workers
        
    Returns:
        Worker ID
    """
    if isinstance(feature_id, str):
        # Convert string to integer using built-in hash
        feature_int = abs(hash(feature_id))
    else:
        feature_int = int(feature_id)
    
    return feature_int % num_workers


def murmur_hash(feature_id: Union[int, str], num_workers: int, seed: int = 42) -> int:
    """
    MurmurHash-based consistent hashing (requires mmh3 package).
    This is more efficient than SHA-256 but requires additional dependency.
    
    Args:
        feature_id: The feature ID to hash
        num_workers: Total number of workers
        seed: Hash seed for reproducibility
        
    Returns:
        Worker ID
    """
    try:
        import mmh3
        feature_str = str(feature_id)
        hash_value = mmh3.hash(feature_str, seed)
        return abs(hash_value) % num_workers
    except ImportError:
        # Fallback to consistent_hash if mmh3 is not available
        return consistent_hash(feature_id, num_workers)


def range_based_hash(feature_id: int, num_workers: int, 
                    feature_range: tuple = None) -> int:
    """
    Range-based hashing for integer feature IDs.
    Useful when feature IDs have known ranges.
    
    Args:
        feature_id: Integer feature ID
        num_workers: Total number of workers
        feature_range: Tuple of (min_feature_id, max_feature_id)
        
    Returns:
        Worker ID
    """
    if not isinstance(feature_id, int):
        raise ValueError("feature_id must be an integer for range-based hashing")
    
    if feature_range is None:
        # Fallback to consistent hashing
        return consistent_hash(feature_id, num_workers)
    
    min_id, max_id = feature_range
    range_size = max_id - min_id + 1
    features_per_worker = range_size // num_workers
    
    if features_per_worker == 0:
        return consistent_hash(feature_id, num_workers)
    
    normalized_id = feature_id - min_id
    worker_id = min(normalized_id // features_per_worker, num_workers - 1)
    
    return worker_id


# Hash function registry for easy switching
HASH_FUNCTIONS = {
    'consistent': consistent_hash,
    'modulo': simple_modulo_hash,
    'murmur': murmur_hash,
    'range': range_based_hash
}


def get_hash_function(name: str):
    """Get hash function by name."""
    return HASH_FUNCTIONS.get(name.lower(), consistent_hash) 