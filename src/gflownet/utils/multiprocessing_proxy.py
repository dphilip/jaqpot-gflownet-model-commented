"""
Multiprocessing proxy utilities for sharing objects across PyTorch DataLoader workers.

This module provides a sophisticated multiprocessing system that allows expensive objects
(like neural network models or large datasets) to be shared across multiple worker processes
without duplicating them in memory. This is particularly useful for PyTorch DataLoaders
where each worker typically gets its own copy of objects.

The system uses a proxy pattern where:
1. The real object lives in the main process
2. Worker processes get placeholder objects that forward method calls
3. Method calls are sent via multiprocessing queues
4. Results are returned back to the calling worker

This approach saves memory and allows centralized control of GPU resources while still
enabling parallel data processing.
"""

# Standard library imports for multiprocessing and serialization
import pickle
import queue
import threading
import traceback

# PyTorch imports for tensor operations and multiprocessing
import torch
import torch.multiprocessing as mp


class MPObjectPlaceholder:
    """
    Placeholder object that forwards method calls to the main process via message queues.
    
    This class acts as a proxy for the real object in worker processes. When methods
    are called on this placeholder, it serializes the call parameters and sends them
    to the main process via multiprocessing queues. The main process executes the
    method on the real object and returns the result.
    
    The placeholder handles:
    - Automatic worker ID detection for queue routing
    - Optional message pickling for memory efficiency
    - Exception propagation from main process to workers
    - Special method forwarding (__call__, __len__, etc.)
    """

    def __init__(self, in_queues, out_queues, pickle_messages=False):
        """
        Initialize the placeholder with communication queues.
        
        Parameters
        ----------
        in_queues : List[mp.Queue]
            Queues for sending messages from workers to main process
            One queue per worker plus one for main process
        out_queues : List[mp.Queue]
            Queues for receiving results from main process to workers
            One queue per worker plus one for main process
        pickle_messages : bool, default=False
            Whether to pickle messages for reduced memory usage
            Trades CPU time for memory efficiency
        """
        self.qs = in_queues, out_queues          # Store queue references
        self.device = torch.device("cpu")        # Default device for tensor operations
        self.pickle_messages = pickle_messages   # Message serialization flag
        self._is_init = False                    # Lazy initialization flag

    def _check_init(self):
        """
        Lazily initialize worker-specific queue connections.
        
        This method determines which worker process is calling and sets up
        the appropriate input/output queues for communication with the main process.
        The initialization is deferred until first use to avoid issues with
        process forking and queue inheritance.
        """
        # Skip if already initialized
        if self._is_init:
            return
            
        # Get worker information from PyTorch DataLoader
        info = torch.utils.data.get_worker_info()
        
        if info is None:
            # Running in main process - use last queue (reserved for main)
            self.in_queue = self.qs[0][-1]
            self.out_queue = self.qs[1][-1]
        else:
            # Running in worker process - use worker-specific queue
            self.in_queue = self.qs[0][info.id]
            self.out_queue = self.qs[1][info.id]
            
        self._is_init = True

    def encode(self, m):
        """
        Encode message for transmission between processes.
        
        Parameters
        ----------
        m : Any
            Message to encode (method name, args, kwargs tuple)
            
        Returns
        -------
        bytes or Any
            Pickled message if pickle_messages=True, otherwise original message
        """
        if self.pickle_messages:
            return pickle.dumps(m)
        return m

    def decode(self, m):
        """
        Decode message received from main process.
        
        Parameters
        ----------
        m : bytes or Any
            Message to decode (pickled if pickle_messages=True)
            
        Returns
        -------
        Any
            Decoded message result
            
        Raises
        ------
        Exception
            If the decoded message is an exception, re-raises it in worker process
        """
        # Unpickle if necessary
        if self.pickle_messages:
            m = pickle.loads(m)
            
        # Check if main process sent an exception
        if isinstance(m, Exception):
            print("Received exception from main process, reraising.")
            raise m
            
        return m

    def __getattr__(self, name):
        """
        Create method wrappers for attribute access on the proxied object.
        
        This method is called when accessing any attribute that doesn't exist
        on the placeholder itself. It creates a wrapper function that forwards
        the method call to the main process.
        
        Parameters
        ----------
        name : str
            Name of the attribute/method being accessed
            
        Returns
        -------
        callable
            Wrapper function that forwards calls to main process
        """
        def method_wrapper(*a, **kw):
            # Ensure queue connections are initialized
            self._check_init()
            # Send method call to main process
            self.in_queue.put(self.encode((name, a, kw)))
            # Wait for and return the result
            return self.decode(self.out_queue.get())

        return method_wrapper

    def __call__(self, *a, **kw):
        """
        Handle direct calls to the placeholder (for callable objects).
        
        This allows the placeholder to be used as a function if the
        underlying object is callable.
        """
        self._check_init()
        self.in_queue.put(self.encode(("__call__", a, kw)))
        return self.decode(self.out_queue.get())

    def __len__(self):
        """
        Handle len() calls on the placeholder.
        
        Returns the length of the underlying object without needing
        to transfer the entire object to the worker process.
        """
        self._check_init()
        self.in_queue.put(("__len__", (), {}))
        return self.out_queue.get()


class MPObjectProxy:
    """This class maintains a reference to some object and
    creates a `placeholder` attribute which can be safely passed to
    multiprocessing DataLoader workers.

    The placeholders in each process send messages accross multiprocessing
    queues which are received by this proxy instance. The proxy instance then
    runs the calls on our object and sends the return value back to the worker.

    Starts its own (daemon) thread.
    Always passes CPU tensors between processes.
    """

    def __init__(self, obj, num_workers: int, cast_types: tuple, pickle_messages: bool = False):
        """Construct a multiprocessing object proxy.

        Parameters
        ----------
        obj: any python object to be proxied (typically a torch.nn.Module or ReplayBuffer)
            Lives in the main process to which method calls are passed
        num_workers: int
            Number of DataLoader workers
        cast_types: tuple
            Types that will be cast to cuda when received as arguments of method calls.
            torch.Tensor is cast by default.
        pickle_messages: bool
            If True, pickle messages sent between processes. This reduces load on shared
            memory, but increases load on CPU. It is recommended to activate this flag if
            encountering "Too many open files"-type errors.
        """
        self.in_queues = [mp.Queue() for i in range(num_workers + 1)]  # type: ignore
        self.out_queues = [mp.Queue() for i in range(num_workers + 1)]  # type: ignore
        self.pickle_messages = pickle_messages
        self.placeholder = MPObjectPlaceholder(self.in_queues, self.out_queues, pickle_messages)
        self.obj = obj
        if hasattr(obj, "parameters"):
            self.device = next(obj.parameters()).device
        else:
            self.device = torch.device("cpu")
        self.cuda_types = (torch.Tensor,) + cast_types
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def encode(self, m):
        if self.pickle_messages:
            return pickle.dumps(m)
        return m

    def decode(self, m):
        if self.pickle_messages:
            return pickle.loads(m)
        return m

    def to_cpu(self, i):
        return i.detach().to(torch.device("cpu")) if isinstance(i, self.cuda_types) else i

    def run(self):
        timeouts = 0

        while not self.stop.is_set() or timeouts < 500:
            for qi, q in enumerate(self.in_queues):
                try:
                    r = self.decode(q.get(True, 1e-5))
                except queue.Empty:
                    timeouts += 1
                    continue
                except ConnectionError:
                    break
                timeouts = 0
                attr, args, kwargs = r
                f = getattr(self.obj, attr)
                args = [i.to(self.device) if isinstance(i, self.cuda_types) else i for i in args]
                kwargs = {k: i.to(self.device) if isinstance(i, self.cuda_types) else i for k, i in kwargs.items()}
                try:
                    # There's no need to compute gradients, since we can't transfer them back to the worker
                    with torch.no_grad():
                        result = f(*args, **kwargs)
                except Exception as e:
                    result = e
                    exc_str = traceback.format_exc()
                    try:
                        pickle.dumps(e)
                    except Exception:
                        result = RuntimeError("Exception raised in MPModelProxy, but it cannot be pickled.\n" + exc_str)
                if isinstance(result, (list, tuple)):
                    msg = [self.to_cpu(i) for i in result]
                elif isinstance(result, dict):
                    msg = {k: self.to_cpu(i) for k, i in result.items()}
                else:
                    msg = self.to_cpu(result)
                self.out_queues[qi].put(self.encode(msg))

    def terminate(self):
        self.stop.set()


def mp_object_wrapper(obj, num_workers, cast_types, pickle_messages: bool = False):
    """Construct a multiprocessing object proxy for torch DataLoaders so
    that it does not need to be copied in every worker's memory. For example,
    this can be used to wrap a model such that only the main process makes
    cuda calls by forwarding data through the model, or a replay buffer
    such that the new data is pushed in from the worker processes but only the
    main process has to hold the full buffer in memory.
                    self.out_queues[qi].put(self.encode(msg))
                elif isinstance(result, dict):
                    msg = {k: self.to_cpu(i) for k, i in result.items()}
                    self.out_queues[qi].put(self.encode(msg))
                else:
                    msg = self.to_cpu(result)
                    self.out_queues[qi].put(self.encode(msg))

    Parameters
    ----------
    obj: any python object to be proxied (typically a torch.nn.Module or ReplayBuffer)
            Lives in the main process to which method calls are passed
    num_workers: int
        Number of DataLoader workers
    cast_types: tuple
        Types that will be cast to cuda when received as arguments of method calls.
        torch.Tensor is cast by default.
    pickle_messages: bool
            If True, pickle messages sent between processes. This reduces load on shared
            memory, but increases load on CPU. It is recommended to activate this flag if
            encountering "Too many open files"-type errors.

    Returns
    -------
    placeholder: MPObjectPlaceholder
        A placeholder object whose method calls route arguments to the main process

    """
    return MPObjectProxy(obj, num_workers, cast_types, pickle_messages)
