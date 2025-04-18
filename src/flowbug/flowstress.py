#!/usr/bin/env python3
import time
import argparse
import concurrent.futures

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Optional: use webdriver-manager to auto‐install chromedriver
from webdriver_manager.chrome import ChromeDriverManager


def parse_args():
    parser = argparse.ArgumentParser(description="Flow Stress Test")
    parser.add_argument(
        "-url",
        "--url",
        type=str,
        default="https://go-flow.co",
        help="URL to test (default: https://go-flow.co)",
    )
    parser.add_argument(
        "-file",
        "--file",
        type=str,
        default="welcome.flowco",
        help="File to test (default: welcome.flowco)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of concurrent sessions to run (default: 1)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of times to run the Run/Cancel loop per session (default: 5)",
    )
    args = parser.parse_args()
    return args.url, args.file, args.n, args.iterations


def run_session(session_id: int, url: str, file: str, iterations: int):
    print(f"[Session {session_id}] Launching Chrome…")
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Run in headless mode
        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()), options=options
        )
    except WebDriverException as e:
        print(f"[Session {session_id}] ERROR launching Chrome: {e}")
        return

    wait = WebDriverWait(driver, 10)  # 10s timeout
    long_wait = WebDriverWait(driver, 180)  # 3m timeout

    try:
        print(f"[Session {session_id}] Opening URL: {url}")
        driver.get(url)

        # Sign in as guest
        login_btn = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[normalize-space()='Sign In As Guest']")
            )
        )
        login_btn.click()

        # Close modal
        close_btn = long_wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Close']"))
        )
        close_btn.click()

        # Projects page
        print(f"[Session {session_id}] Navigating to projects_main")
        driver.get(f"{url}/projects_main")

        time.sleep(5)

        # Select the file
        print(f"[Session {session_id}] Selecting file: {file}")
        file_btn = wait.until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    f"//div[contains(normalize-space(), '{file.split('.')[0]}')]/ancestor::button",
                )
            )
        )
        file_btn.click()

        time.sleep(5)

        # Click “Edit”
        edit_btn = long_wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//a[contains(normalize-space(), 'Edit')]")
            )
        )
        edit_btn.click()

        time.sleep(5)

        # Run/Cancel loop
        for i in range(1, iterations + 1):
            time.sleep(2)
            try:
                try:
                    print(
                        f"[Session {session_id}] Iteration {i}/{iterations} — clicking Run"
                    )
                    run_btn = wait.until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "//button[contains(normalize-space(), 'Run')]",
                            )
                        )
                    )
                    run_btn.click()
                    time.sleep(0.5)
                except Exception as e:
                    print(f"[Session {session_id}] Iteration {i}: {e}")

                try:
                    print(
                        f"[Session {session_id}] Iteration {i}/{iterations} — clicking Cancel"
                    )
                    # Wait for the Cancel button to be clickable
                    cancel_btn = long_wait.until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "//button[contains(normalize-space(), 'cancel_presentation')]",
                            )
                        )
                    )
                    cancel_btn.click()
                except Exception as e:
                    print(f"[Session {session_id}] Iteration {i}: {e}")

                # cancel_btn = long_wait.until(
                #     EC.element_to_be_clickable(
                #         (
                #             By.XPATH,
                #             "//button[contains(normalize-space(), 'cancel_presentation')]",
                #         )
                #     )
                # )
                # cancel_btn.click()

                # Short pause before next iteration
                time.sleep(1)

            except TimeoutException as te:
                print(
                    f"[Session {session_id}] Iteration {i}: element wait timed out: {te}"
                )
            except Exception as ex:
                print(f"[Session {session_id}] Iteration {i}: unexpected error: {ex}")

    except Exception as e:
        print(f"[Session {session_id}] ERROR: {e}")

    finally:
        print(f"[Session {session_id}] Quitting Chrome")
        driver.quit()


def main():
    url, file, n, iterations = parse_args()

    if n <= 1:
        run_session(1, url, file, iterations)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executer:
            futures = [
                executer.submit(run_session, sid, url, file, iterations)
                for sid in range(1, n + 1)
            ]
            concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()
