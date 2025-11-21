"""
Parameter Server Client for distributed PyTorch training.
Provides client-side API for workers to communicate with parameter servers.
"""

import logging
import requests
import json
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import time

from utils.hashing import consistent_hash


logger = logging.getLogger(__name__)


class ParameterServerClient:
    """Client for communicating with parameter servers."""
    
    def __init__(self, driver_host: str, driver_port: int, 
                 worker_hosts: List[str], worker_port_base: int,
                 timeout: float = 30.0, max_retries: int = 3):
        """
        Initialize parameter server client.
        
        Args:
            driver_host: Hostname of the driver parameter server
            driver_port: Port of the driver parameter server
            worker_hosts: List of worker hostnames
            worker_port_base: Base port for worker parameter servers
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.driver_host = driver_host
        self.driver_port = driver_port
        self.worker_hosts = worker_hosts
        self.worker_port_base = worker_port_base
        self.timeout = timeout
        self.max_retries = max_retries
        self.num_workers = len(worker_hosts)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Thread pool for concurrent requests
        self.executor = ThreadPoolExecutor(max_workers=min(32, self.num_workers * 2))
        
        logger.info(f"Parameter server client initialized with {self.num_workers} workers")
    
    def fetch_model_from_driver(self) -> Dict[str, torch.Tensor]:
        """
        Fetch the current main model parameters from the driver.
        
        Returns:
            Dictionary mapping parameter names to tensors
        """
        url = f"http://{self.driver_host}:{self.driver_port}/get_model"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                if data['status'] != 'success':
                    raise RuntimeError(f"Driver returned error: {data.get('message', 'Unknown error')}")
                
                # Convert serialized model state back to tensors
                model_state = {}
                for name, tensor_data in data['model_state'].items():
                    model_state[name] = torch.tensor(tensor_data)
                
                logger.debug(f"Fetched model with {len(model_state)} parameters from driver")
                return model_state
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed to fetch model from driver: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to fetch model from driver after {self.max_retries} attempts")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def fetch_embeddings_from_workers(self, feature_ids: List[int]) -> Dict[int, torch.Tensor]:
        """
        Fetch embeddings for the given feature IDs from appropriate workers.
        
        Args:
            feature_ids: List of feature IDs to fetch
            
        Returns:
            Dictionary mapping feature IDs to embedding tensors
        """
        if not feature_ids:
            return {}
        
        # Group feature IDs by the worker that owns them
        worker_requests = self._group_features_by_worker(feature_ids)
        
        # Fetch embeddings concurrently from all relevant workers
        embeddings = {}
        futures = []
        
        for worker_id, worker_feature_ids in worker_requests.items():
            future = self.executor.submit(
                self._fetch_embeddings_from_worker, 
                worker_id, 
                worker_feature_ids
            )
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                worker_embeddings = future.result()
                embeddings.update(worker_embeddings)
            except Exception as e:
                logger.error(f"Failed to fetch embeddings from worker: {str(e)}")
                raise
        
        logger.debug(f"Fetched {len(embeddings)} embeddings from {len(worker_requests)} workers")
        return embeddings
    
    def push_gradients_to_driver(self, gradients: Dict[str, torch.Tensor]):
        """
        Push gradients for the main model to the driver parameter server.
        
        Args:
            gradients: Dictionary mapping parameter names to gradient tensors
        """
        if not gradients:
            return
        
        url = f"http://{self.driver_host}:{self.driver_port}/update_gradients"
        
        # Serialize gradients
        serialized_gradients = {}
        for name, grad_tensor in gradients.items():
            serialized_gradients[name] = grad_tensor.tolist()
        
        payload = {'gradients': serialized_gradients}
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url, 
                    data=json.dumps(payload), 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                if data['status'] != 'success':
                    raise RuntimeError(f"Driver returned error: {data.get('message', 'Unknown error')}")
                
                logger.debug(f"Pushed {len(gradients)} gradients to driver")
                return
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed to push gradients to driver: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to push gradients to driver after {self.max_retries} attempts")
                time.sleep(2 ** attempt)
    
    def push_embedding_gradients_to_workers(self, embedding_gradients: Dict[int, torch.Tensor]):
        """
        Push embedding gradients to the appropriate worker parameter servers.
        
        Args:
            embedding_gradients: Dictionary mapping feature IDs to gradient tensors
        """
        if not embedding_gradients:
            return
        
        # Group gradients by the worker that owns them
        worker_gradients = self._group_gradients_by_worker(embedding_gradients)
        
        # Push gradients concurrently to all relevant workers
        futures = []
        
        for worker_id, gradients in worker_gradients.items():
            future = self.executor.submit(
                self._push_gradients_to_worker,
                worker_id,
                gradients
            )
            futures.append(future)
        
        # Wait for all pushes to complete
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                logger.error(f"Failed to push gradients to worker: {str(e)}")
                raise
        
        logger.debug(f"Pushed gradients for {len(embedding_gradients)} embeddings to {len(worker_gradients)} workers")
    
    def get_driver_model_info(self) -> Dict[str, Any]:
        """Get information about the driver model."""
        url = f"http://{self.driver_host}:{self.driver_port}/get_model_state"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get driver model info: {str(e)}")
            raise
    
    def get_worker_shard_info(self, worker_id: int) -> Dict[str, Any]:
        """Get information about a worker's embedding shard."""
        host = self.worker_hosts[worker_id]
        port = self.worker_port_base + worker_id
        url = f"http://{host}:{port}/get_shard_info"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get worker {worker_id} shard info: {str(e)}")
            raise
    
    def health_check_all(self) -> Dict[str, bool]:
        """Check health of all parameter servers."""
        results = {"driver": False}
        
        # Check driver
        try:
            url = f"http://{self.driver_host}:{self.driver_port}/health"
            response = self.session.get(url, timeout=5.0)
            results["driver"] = response.status_code == 200
        except:
            pass
        
        # Check workers
        for worker_id in range(self.num_workers):
            try:
                host = self.worker_hosts[worker_id]
                port = self.worker_port_base + worker_id
                url = f"http://{host}:{port}/health"
                response = self.session.get(url, timeout=5.0)
                results[f"worker_{worker_id}"] = response.status_code == 200
            except:
                results[f"worker_{worker_id}"] = False
        
        return results
    
    def _group_features_by_worker(self, feature_ids: List[int]) -> Dict[int, List[int]]:
        """Group feature IDs by the worker that owns them."""
        worker_requests = {}
        
        for feature_id in feature_ids:
            worker_id = consistent_hash(feature_id, self.num_workers)
            if worker_id not in worker_requests:
                worker_requests[worker_id] = []
            worker_requests[worker_id].append(feature_id)
        
        return worker_requests
    
    def _group_gradients_by_worker(self, embedding_gradients: Dict[int, torch.Tensor]) -> Dict[int, Dict[str, List]]:
        """Group embedding gradients by the worker that owns them."""
        worker_gradients = {}
        
        for feature_id, grad_tensor in embedding_gradients.items():
            worker_id = consistent_hash(feature_id, self.num_workers)
            if worker_id not in worker_gradients:
                worker_gradients[worker_id] = {}
            worker_gradients[worker_id][str(feature_id)] = grad_tensor.tolist()
        
        return worker_gradients
    
    def _fetch_embeddings_from_worker(self, worker_id: int, feature_ids: List[int]) -> Dict[int, torch.Tensor]:
        """Fetch embeddings from a specific worker."""
        host = self.worker_hosts[worker_id]
        port = self.worker_port_base + worker_id
        url = f"http://{host}:{port}/get_embeddings"
        
        payload = {'feature_ids': feature_ids}
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                if data['status'] != 'success':
                    raise RuntimeError(f"Worker {worker_id} returned error: {data.get('message', 'Unknown error')}")
                
                # Convert embeddings back to tensors
                embeddings = {}
                for feature_id_str, embedding_data in data['embeddings'].items():
                    feature_id = int(feature_id_str)
                    embeddings[feature_id] = torch.tensor(embedding_data)
                
                return embeddings
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed to fetch embeddings from worker {worker_id}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to fetch embeddings from worker {worker_id} after {self.max_retries} attempts")
                time.sleep(2 ** attempt)
    
    def _push_gradients_to_worker(self, worker_id: int, gradients: Dict[str, List]):
        """Push gradients to a specific worker."""
        host = self.worker_hosts[worker_id]
        port = self.worker_port_base + worker_id
        url = f"http://{host}:{port}/update_embedding_gradients"
        
        payload = {'gradients': gradients}
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                if data['status'] != 'success':
                    raise RuntimeError(f"Worker {worker_id} returned error: {data.get('message', 'Unknown error')}")
                
                return
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed to push gradients to worker {worker_id}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to push gradients to worker {worker_id} after {self.max_retries} attempts")
                time.sleep(2 ** attempt)
    
    def close(self):
        """Close the client and clean up resources."""
        self.executor.shutdown(wait=True)
        self.session.close()
        logger.info("Parameter server client closed") 