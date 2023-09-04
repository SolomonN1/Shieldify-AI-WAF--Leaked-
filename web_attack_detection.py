from ml.Detection.xss import XSSDetector
from ml.Detection.sqli import SQLiDetector
from ml.Detection.lfi import LFIDetector
from ml.utils.filenames import (
    xss_d2v_model_path,
    xss_ml_model_paths,
    lfi_d2v_model_path,
    lfi_ml_model_paths,
    sqli_d2v_model_path,
    sqli_ml_model_paths,
)
from concurrent.futures import ThreadPoolExecutor
import threading

class DetectWebAttack:
    def __init__(self, text):
        self.text = text
        self.lock = threading.Lock()  # Lock for synchronization

    def xss_detection(self):
        test_xss = self.text
        detector = XSSDetector(xss_d2v_model_path, xss_ml_model_paths)
        return detector.classify_xss(test_xss)

    def lfi_detection(self):
        test_lfi = self.text
        detector = LFIDetector(lfi_d2v_model_path, lfi_ml_model_paths)
        return detector.classify_lfi(test_lfi)

    def sqli_detection(self):
        test_sqli = self.text
        detector = SQLiDetector(sqli_d2v_model_path, sqli_ml_model_paths)
        return detector.classify_sqli(test_sqli)

def process_detection(detection, attack_type):
    result = getattr(detection, f"{attack_type}_detection")()
    with detection.lock:
        print(f"{attack_type.upper()} Detection Result: {result}")
        return result

def perform_web_attack_detection(attack):
    test = [attack]
    detection = DetectWebAttack(test)

    with ThreadPoolExecutor() as executor:
        # Submit detection tasks
        xss_future = executor.submit(process_detection, detection, "xss")
        lfi_future = executor.submit(process_detection, detection, "lfi")
        sqli_future = executor.submit(process_detection, detection, "sqli")

    # Get the results from the detection futures
    xss_result = xss_future.result()
    lfi_result = lfi_future.result()
    sqli_result = sqli_future.result()

    # Check if any of the results are [True]
    if [True] in (xss_result, lfi_result, sqli_result):
        return True
    else:
        return False