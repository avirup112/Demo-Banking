"""
Custom logger for Debt collection ML project
Provides structured, contextual logging with performance monitoring
"""

import logging 
import logging.handlers
import json
import sys 
import os 
import time 
import traceback 
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from functools import wraps
import threading 
from contextlib import contextmanager

class DebtCollectionMLLogger:
    """
    Enhanced logger for debt collection ML pipeline with:
    - Structured logging with context
    - Performance monitoring 
    - MLOps integration 
    - Component-specific logging 
    - Audit trail for model decisions
    """
    
    _instances = {}
    _lock = threading.Lock()

    # component-specific log levels 
    COMPONENT_LEVELS = {
        'data': logging.INFO,
        'preprocessing': logging.INFO,
        'features': logging.INFO,
        'training': logging.INFO,
        'validation': logging.INFO,
        'monitoring': logging.WARNING,
        'drift': logging.WARNING,
        'ab_testing': logging.INFO,
        'deployment': logging.INFO,
        'api': logging.INFO,
        'dagshub': logging.INFO,
        'performance': logging.DEBUG,
        'audit': logging.INFO
    }

    def __new__(cls, component: str = 'main', log_dir: str = None):
        """singleton pattern per component"""
        with cls._lock:
            if component not in cls._instances:
                cls._instances[component] = super().__new__(cls)
                cls._instances[component]._initialized = False
            return cls._instances[component]

    def __init__(self, component: str = 'main', log_dir: str = None):
        """Intialize logger for specific component"""
        if self._initialized:
            return
        
        self.component = component
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Performance tracking 
        self.performance_data = {}
        self.context_stack = []

        # setup loggers
        self._steup_loggers()
        self._intialized = True

    def _setup_loggers(self):
        """Setup multiple loggers for different purposes"""

        # Main application logger
        self.logger = logging.getLogger(f'debt_collection_ml.{self.component}')
        self.logger.setLevel(self.COMPONENT_LEVELS.get(self.component, logging.INFO))

        # clear existing handlers
        self.logger.handlers.clear()

        # console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(component)s | %(levelname)s | %(message)s',
            datefmt = '%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler for detailed logs
        log_file = self.log_dir / f"{self.component}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.haandlers.RotatingFileHandler(
            log_file, maxBytes = 10*1024*1024, background=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Performance logger
        self.perf_logger = logging.getLogger(f'debt_collection_ml.performance.{self.component}')
        perf_file = self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
        perf_handler = logging.FileHandler(perf_file)
        perf_handler.setFormatter(logging.Formatter('%(message)s'))
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)

        # Audit logger for model decisions and data changes
        self.audit_logger = logging.getLogger(f'debt_collection_ml.audit.{self.component}')
        audit_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        audit_handler = logging.FileHandler(audit_file)
        audit_handler.setFormatter(StructuredFormatter())
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)

        # Error logger for critical issues
        self.error_logger = logging.getLogger(f'debt_collection_ml.errors.{self.component}')
        error_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setFormatter(StructuredFormatter())
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.ERROR)

    def _get_context(self) -> Dict[str, Any]:
        """Get current logging context"""
        context = {
            'component': self.component,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'thread_id': threading.current_thread().ident,
            'process_id': os.getpid()
        }
        
        # Add context stack
        if self.context_stack:
            context['context_stack'] = self.context_stack.copy()
        
        return context

    def info(self, message: str, **kwargs):
        """Log info message with context"""
        extra = self._get_context()
        extra.update(kwargs)
        self.logger.info(message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        extra = self._get_context()
        extra.update(kwargs)
        self.logger.debug(message, extra=extra)

    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        extra = self._get_context()
        extra.update(kwargs)
        self.logger.warning(message, extra=extra)

    def error(self, message: str, exception: Exception = None, **kwargs):
        """Log error message with context and exception details"""
        extra = self._get_context()
        extra.update(kwargs)
        
        if exception:
            extra['exception_type'] = type(exception).__name__
            extra['exception_message'] = str(exception)
            extra['traceback'] = traceback.format_exc()
        
        self.logger.error(message, extra=extra)
        self.error_logger.error(message, extra=extra)

    def critical(self, message: str, exception: Exception = None, **kwargs):
        """Log critical message with context"""
        extra = self._get_context()
        extra.update(kwargs)
        
        if exception:
            extra['exception_type'] = type(exception).__name__
            extra['exception_message'] = str(exception)
            extra['traceback'] = traceback.format_exc()
        
        self.logger.critical(message, extra=extra)
        self.error_logger.critical(message, extra=extra)

    def audit(self, action: str, details: Dict[str, Any] = None, **kwargs):
        """Log audit trail for important actions"""
        extra = self._get_context()
        extra['action'] = action
        extra['audit_type'] = 'action'
        if details:
            extra['details'] = details
        extra.update(kwargs)
        
        message = f"AUDIT: {action}"
        self.audit_logger.info(message, extra=extra)

    def model_decision(self, model_name: str, input_data: Dict[str, Any], prediction: Any, confidence: float = None, **kwargs):
        """Log model prediction for audit trail"""
        extra = self._get_context()
        extra.update({
            'audit_type': 'model_decision',
            'model_name': model_name,
            'input_hash': hash(str(sorted(input_data.items()))),
            'prediction': prediction,
            'confidence': confidence,
            **kwargs
        })
        
        message = f"MODEL_DECISION: {model_name} -> {prediction}"
        self.audit_logger.info(message, extra=extra)

    def data_change(self, operation: str, table_name: str = None, 
                   record_count: int = None, **kwargs):
        """Log data changes for audit trail"""
        extra = self._get_context()
        extra.update({
            'audit_type': 'data_change',
            'operation': operation,
            'table_name': table_name,
            'record_count': record_count,
            **kwargs
        })
        
        message = f"DATA_CHANGE: {operation}"
        if table_name:
            message += f" on {table_name}"
        if record_count:
            message += f" ({record_count} records)"
            
        self.audit_logger.info(message, extra=extra)

    def performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        extra = self._get_context()
        extra.update({
            'operation': operation,
            'duration_seconds': duration,
            'performance_type': 'timing',
            **kwargs
        })
        
        # Store in performance data
        if operation not in self.performance_data:
            self.performance_data[operation] = []
        self.performance_data[operation].append(duration)
        
        message = f"PERF: {operation} completed in {duration:.3f}s"
        self.perf_logger.info(json.dumps(extra))
        self.debug(message, **kwargs)

    def metrics(self, metric_name: str, value: Union[int, float], 
               unit: str = None, **kwargs):
        """Log custom metrics"""
        extra = self._get_context()
        extra.update({
            'metric_name': metric_name,
            'metric_value': value,
            'metric_unit': unit,
            'performance_type': 'metric',
            **kwargs
        })
        
        message = f"METRIC: {metric_name} = {value}"
        if unit:
            message += f" {unit}"
            
        self.perf_logger.info(json.dumps(extra))
        self.debug(message, **kwargs)

    @contextmanager
    def context(self, context_name: str, **context_data):
        """Context manager for nested logging context"""
        context_info = {'name': context_name, **context_data}
        self.context_stack.append(context_info)
        
        start_time = time.time()
        self.debug(f"Entering context: {context_name}", **context_data)
        
        try:
            yield self
        except Exception as e:
            self.error(f"Error in context {context_name}", exception=e, **context_data)
            raise
        finally:
            duration = time.time() - start_time
            self.debug(f"Exiting context: {context_name} (duration: {duration:.3f}s)", 
                      duration=duration, **context_data)
            self.context_stack.pop()
    
    def timer(self, operation_name: str = None):
        """Decorator for timing function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                start_time = time.time()
                
                try:
                    self.debug(f"Starting {op_name}")
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.performance(op_name, duration)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.error(f"Failed {op_name} after {duration:.3f}s", exception=e)
                    raise
            return wrapper
        return decorator
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        summary = {}
        for operation, durations in self.performance_data.items():
            if durations:
                summary[operation] = {
                    'count': len(durations),
                    'total_time': sum(durations),
                    'avg_time': sum(durations) / len(durations),
                    'min_time': min(durations),
                    'max_time': max(durations)
                }
        return summary
    
    def log_performance_summary(self):
        """Log performance summary"""
        summary = self.get_performance_summary()
        if summary:
            self.info("Performance Summary", performance_summary=summary)
    
    @classmethod
    def get_logger(cls, component: str = 'main', log_dir: str = None) -> 'DebtCollectionMLLogger':
        """Get logger instance for component"""
        return cls(component, log_dir)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add component to record
        if hasattr(record, 'component'):
            record.component = record.component
        else:
            record.component = 'main'
            
        # Add color
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname}{self.RESET}"
        
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'component': getattr(record, 'component', 'main'),
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


# Convenience functions for easy usage
def get_logger(component: str = 'main', log_dir: str = None) -> DebtCollectionMLLogger:
    """Get logger instance for component"""
    return DebtCollectionMLLogger.get_logger(component, log_dir)


# Component-specific logger getters
def get_data_logger(log_dir: str = None) -> DebtCollectionMLLogger:
    """Get logger for data processing components"""
    return get_logger('data', log_dir)


def get_training_logger(log_dir: str = None) -> DebtCollectionMLLogger:
    """Get logger for model training components"""
    return get_logger('training', log_dir)


def get_monitoring_logger(log_dir: str = None) -> DebtCollectionMLLogger:
    """Get logger for monitoring components"""
    return get_logger('monitoring', log_dir)


def get_api_logger(log_dir: str = None) -> DebtCollectionMLLogger:
    """Get logger for API components"""
    return get_logger('api', log_dir)
    
    
