import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, tb = error_detail.exc_info()
    if tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        lineno = tb.tb_lineno
        error_message = f"Error in [{filename}] at line number:[{lineno}] For more info: {error}"
    else:
        error_message = f"Error: {error}"

    logging.error(error_message)
    return error_message

class MyException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
