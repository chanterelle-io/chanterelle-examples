"""
Base handler for ML models that provides communication protocol.
Rust calls this file directly, and this imports user functions from handler_io.py
"""
import json
import sys
import importlib.util
import os
import inspect
import traceback
import logging
from typing import Any, Dict, Optional, TextIO

# Protocol IO isolation: keep a dedicated handle to original stdout for JSON messages
_PROTOCOL_OUT: Optional[TextIO] = None
_IO_ISOLATED: bool = False


def _setup_io_isolation() -> None:
    """Reserve stdout for protocol JSON only; send all other output to stderr.

    - Duplicate original stdout (fd=1) and keep it for protocol writes.
    - Redirect fd=1 to fd=2 so accidental prints/C-level writes go to stderr.
    - Point sys.stdout to sys.stderr for Python-level print().
    - Reduce noisy ML logs and set logging to WARNING to stderr.
    """
    global _PROTOCOL_OUT, _IO_ISOLATED
    if _IO_ISOLATED:
        return

    # Reduce noise from TF before imports
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        orig_stdout_fd = os.dup(1)
        _PROTOCOL_OUT = os.fdopen(orig_stdout_fd, "w", buffering=1, encoding="utf-8")
        os.dup2(2, 1)  # Redirect OS-level stdout to stderr
        sys.stdout = sys.stderr  # Python print -> stderr
    except Exception:
        # Fallback: at least keep a handle to original stdout
        _PROTOCOL_OUT = sys.__stdout__

    try:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    except Exception:
        pass

    _IO_ISOLATED = True


def _send_protocol_json(obj: Dict[str, Any]) -> None:
    """Write a compact JSON line to the preserved protocol stdout pipe."""
    out = _PROTOCOL_OUT or sys.__stdout__
    out.write(json.dumps(obj, separators=(",", ":")) + "\n")
    out.flush()


def load_user_handler_module(handler_path: str):
    """Load the user's handler_io.py module dynamically."""
    spec = importlib.util.spec_from_file_location("handler_io", handler_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load handler from {handler_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_detailed_error(exception: Exception, context: str = "") -> dict:
    """
    Format a detailed error message with line numbers and stack trace.
    
    Args:
        exception: The caught exception
        context: Additional context about where the error occurred
        
    Returns:
        Dictionary with detailed error information
    """
    # Get the current exception info
    exc_type, exc_value, exc_tb = sys.exc_info()
    
    # Format the basic error message
    error_msg = str(exception)
    error_type = type(exception).__name__
    
    # Extract line number and file information from traceback
    line_info = []
    if exc_tb:
        tb_list = traceback.extract_tb(exc_tb)
        frame_count = 0
        max_frames = 10  # Limit to 10 frames for safety
        
        for frame in tb_list:
            # Only include frames from user code (handler_io.py) or imports
            if 'python_handler_base.py' not in frame.filename:
                line_info.append({
                    "file": os.path.basename(frame.filename),
                    "line": frame.lineno,
                    "function": frame.name,
                    "code": frame.line.strip() if frame.line else "N/A"
                })
                frame_count += 1
                if frame_count >= max_frames:
                    break
    
    # Create detailed error response
    detailed_error = {
        "error": error_msg,
        "error_type": error_type,
        "context": context,
        "traceback": line_info
    }
    
    # Add a formatted summary for easy reading
    if line_info:
        last_frame = line_info[-1]
        detailed_error["summary"] = f"{error_type}: {error_msg} (in {last_frame['file']}:{last_frame['line']})"
    else:
        detailed_error["summary"] = f"{error_type}: {error_msg}"
    
    return detailed_error


def format_simple_error_with_line(exception: Exception) -> dict:
    """
    Format a simple error message with just the line number.
    
    Args:
        exception: The caught exception
        
    Returns:
        Dictionary with error message including line number
    """
    # Get the current exception info
    exc_type, exc_value, exc_tb = sys.exc_info()
    
    # Format the basic error message
    error_msg = str(exception)
    
    # Extract line number from traceback
    line_info = None
    if exc_tb:
        tb_list = traceback.extract_tb(exc_tb)
        # Look for line numbers from user code first, then any code
        for frame in reversed(tb_list):
            if 'handler_io.py' in frame.filename:
                line_info = f"line {frame.lineno} in {os.path.basename(frame.filename)}"
                break
        
        # If no handler_io.py found, try to get line from any frame (fallback)
        if not line_info and tb_list:
            # Get the last frame that has a line number
            for frame in reversed(tb_list):
                if frame.lineno > 0:  # Make sure we have a valid line number
                    line_info = f"line {frame.lineno}"
                    break
    
    # Create simple error response with line number
    if line_info:
        error_with_line = f"{error_msg} ({line_info})"
    else:
        error_with_line = error_msg
    
    return {"error": error_with_line}


class MLModelHandler:
    """
    ML Model Handler that follows SageMaker pattern.
    Imports and calls user's: model_fn, input_fn, predict_fn, output_fn
    """
    
    def __init__(self, handler_module_path: str):
        self.model = None
        self.is_initialized = False
        self.handler_module = load_user_handler_module(handler_module_path)
        self.additional_resources = {}  # Store additional loaded resources
    
    def _call_user_function_with_optional_resources(self, func_name: str, *args, **kwargs):
        """
        Call a user function, automatically determining if it accepts resources parameter.
        This ensures backward compatibility with existing handlers.
        """
        if not hasattr(self.handler_module, func_name):
            return None
        
        func = getattr(self.handler_module, func_name)
        sig = inspect.signature(func)
        
        # Check if function accepts a 'resources' parameter
        accepts_resources = 'resources' in sig.parameters
        
        if accepts_resources:
            # Call with resources
            return func(*args, resources=self.additional_resources, **kwargs)
        else:
            # Call without resources (backward compatibility)
            return func(*args, **kwargs)
        
    def initialize(self):
        """Initialize the model using user's model_fn."""
        try:
            if not hasattr(self.handler_module, 'model_fn'):
                return {"status": "error", "error": "Handler must implement model_fn()"}
            
            # Call user's model_fn to load the model
            model_dir = os.path.dirname(self.handler_module.__file__) if hasattr(self.handler_module, '__file__') else '.'
            self.model = self.handler_module.model_fn(model_dir)
            
            # Load additional resources if user provides an init_resources_fn
            if hasattr(self.handler_module, 'init_resources_fn'):
                self.additional_resources = self.handler_module.init_resources_fn(model_dir)
            
            self.is_initialized = True
            
            return {"status": "ready", "message": "Model loaded successfully"}
        except Exception as e:
            detailed_error = format_detailed_error(e, "model initialization")
            return {"status": "error", **detailed_error}

    def health_check(self):
        """Check if the model is ready."""
        try:
            if self.model is not None and self.is_initialized:
                return {"pong": True, "status": "ready"}
            else:
                return {"pong": False, "status": "not_ready", "error": "Model not loaded"}
        except Exception as e:
            detailed_error = format_detailed_error(e, "health check")
            return {"pong": False, "status": "error", **detailed_error}

    def handle_request(self, request_data):
        """Process a request using SageMaker-style functions."""
        if not self.is_initialized or self.model is None:
            return {"error": "Model not initialized"}
        
        try:
            # Parse input if it's a string
            if isinstance(request_data, str):
                original_input = json.loads(request_data)
            else:
                original_input = request_data
            
            # Step 1: Transform input (user's input_fn)
            if hasattr(self.handler_module, 'input_fn'):
                try:
                    processed_input = self._call_user_function_with_optional_resources('input_fn', original_input)
                except Exception as e:
                    return format_detailed_error(e, "input processing (input_fn)")
            else:
                processed_input = original_input
            
            # Step 2: Make prediction (user's predict_fn)
            if hasattr(self.handler_module, 'predict_fn'):
                try:
                    prediction = self._call_user_function_with_optional_resources('predict_fn', processed_input, self.model)
                except Exception as e:
                    return format_detailed_error(e, "model prediction (predict_fn)")
            else:
                return {"error": "Handler must implement predict_fn()"}
            
            # Step 3: Transform output (user's output_fn)
            if hasattr(self.handler_module, 'output_fn'):
                try:
                    result = self._call_user_function_with_optional_resources('output_fn', prediction, original_input)
                except Exception as e:
                    return format_detailed_error(e, "output processing (output_fn)")
            else:
                result = prediction
            
            return result
            
        except Exception as e:
            # Catch-all for any other errors (like JSON parsing)
            return format_detailed_error(e, "request handling")

    def run_communication_loop(self):
        """Run the main communication loop for stdin/stdout protocol."""
    # IO isolation is performed once at process start
        # Initialize model
        init_result = self.initialize()
        if init_result["status"] != "ready":
            # Emit structured error on protocol stdout for the Rust side to consume
            _send_protocol_json(init_result)
            # Also exit to signal fatal init failure
            sys.exit(1)
        
        print("Model ready. Enter JSON requests (one per line):", file=sys.stderr)
        
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            # Handle health check ping
            if line == '{"ping": true}':
                result = self.health_check()
                _send_protocol_json(result)
                continue
            
            # Process regular requests
            result = self.handle_request(line)
            _send_protocol_json(result)


if __name__ == "__main__":
    # Get the user's handler file path from command line argument
    _setup_io_isolation()
    if len(sys.argv) < 2:
        print("Error: handler_io.py path required as argument", file=sys.stderr)
        sys.exit(1)
    
    handler_path = sys.argv[1]
    if not os.path.exists(handler_path):
        print(f"Error: Handler file not found: {handler_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create and run the handler
    try:
        handler = MLModelHandler(handler_path)
        handler.run_communication_loop()
    except Exception as e:
        # Format detailed error for handler initialization failures
        detailed_error = format_detailed_error(e, "handler initialization")
        error_message = f"Error: Failed to initialize handler: {detailed_error['summary']}"
        
        # Print summary to stderr for immediate visibility
        print(error_message, file=sys.stderr)
        
        # Also print detailed error info to stderr for debugging
        print(f"Detailed error info: {json.dumps(detailed_error, indent=2)}", file=sys.stderr)
        sys.exit(1)
