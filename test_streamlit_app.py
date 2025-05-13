import unittest
import subprocess
import time
import signal
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import NoSuchElementException
import os
import shutil

STREAMLIT_PORT = 8501

# 1) On essaie de trouver geckodriver dans le PATH
GECKO = shutil.which("geckodriver")

# 2) Si pas trouvé, on regarde manuellement
if not GECKO:
    for p in ("/usr/bin/geckodriver", "/usr/local/bin/geckodriver"):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            GECKO = p
            break

# 3) Si toujours pas trouvé, on échoue avec un message clair
if not GECKO:
    raise RuntimeError(
        "Impossible de trouver geckodriver : installe-le (e.g. 'sudo apt install geckodriver') "
        "ou place-le dans /usr/bin/ ou /usr/local/bin."
    )


class TestStreamlitApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1) Démarrage du serveur Streamlit
        cls.proc = subprocess.Popen(
            ["streamlit", "run", "streamlit_app.py", "--server.port", str(STREAMLIT_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 2) Attente dynamique que le serveur soit prêt
        start = time.time()
        while time.time() - start < 10:
            try:
                if requests.get(f"http://localhost:{STREAMLIT_PORT}").status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Le serveur Streamlit n’a pas démarré dans le temps imparti")

        # 3) Configuration du driver Firefox
        options = webdriver.FirefoxOptions()
        options.add_argument("--headless")
        service = Service(GECKO)
        cls.driver = webdriver.Firefox(service=service, options=options)
        cls.driver.get(f"http://localhost:{STREAMLIT_PORT}")

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()
        cls.proc.send_signal(signal.SIGINT)
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()

    def test_prediction_display(self):
        textarea = self.driver.find_element(By.TAG_NAME, "textarea")
        textarea.clear()
        textarea.send_keys("I love this service")
        button = self.driver.find_element(By.XPATH, "//button[contains(.,'Predict')]")
        button.click()

        start = time.time()
        while time.time() - start < 5:
            try:
                result = self.driver.find_element(By.CSS_SELECTOR, "div[data-testid='stAlert-success']")
                break
            except NoSuchElementException:
                time.sleep(0.5)
        else:
            self.fail("Résultat de prédiction non trouvé dans le délai imparti")

        txt = result.text.lower()
        self.assertIn("prediction:", txt)
        self.assertTrue("ham" in txt or "spam" in txt)


if __name__ == '__main__':
    unittest.main()
