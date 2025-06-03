import requests
import os
import time
import subprocess
import json
import concurrent.futures
import threading
import dotenv
from lib.codeparsing_utils import get_venv_env, setup_repo_env
import sys

# --------------------------
# CONFIGURATION CONSTANTS
# --------------------------

dotenv.load_dotenv()
NUM_REPOS_TO_SCRAPE = 1000
MAX_TEST_DURATION = 30
CACHE_FILE = "failed_repos_cache.json"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Global shared state and lock for thread-safe updates.
saved_repo_count = 0
saved_repo_count_lock = threading.Lock()

# Global failed cache (loaded from file).
failed_cache = None


def load_failed_cache():
    global failed_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            if isinstance(cache, list):
                failed_cache = set(cache)
                return failed_cache
        except Exception as e:
            print("Error loading cache:", e)
    failed_cache = set()
    return failed_cache


def save_failed_cache(cache_set):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(list(cache_set), f)
    except Exception as e:
        print("Error saving cache:", e)


def cleanup_repo_dir(repo_dir):
    """
    Delete the repository directory if it exists.
    """
    if os.path.exists(repo_dir):
        print(f"Cleaning up repository directory {repo_dir}...")
        try:
            subprocess.run(["rm", "-rf", repo_dir], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error cleaning up directory {repo_dir}: {e}")


def repo_has_file(owner, repo, filepath):
    """
    Check if a file exists in the repository.
    (Placeholder implementation – replace with your actual logic.)
    """
    return True


def has_non_negligible_tests(owner, repo):
    """
    Check if the repository has a non-negligible test suite.
    (Placeholder implementation – replace with your actual logic.)
    """
    return True


def clone_and_setup_repo(repo):
    """
    Clone the repository, create a virtual environment, install dependencies,
    and run the tests. If any step fails, the repository directory is cleaned up.
    Returns True if all steps succeed and the test duration is acceptable;
    otherwise, returns False.
    """
    owner = repo["owner"]["login"]
    repo_name = repo["name"]
    clone_url = repo["clone_url"]
    full_name = repo["full_name"]

    base_dir = os.path.expanduser(repos_dir)
    repo_dir = os.path.join(base_dir, repo_name)
    os.makedirs(base_dir, exist_ok=True)

    # Clone the repository if it isn't already cloned.
    if not os.path.exists(repo_dir):
        print(f"Cloning {repo_name} into {repo_dir}...")
        try:
            subprocess.run(["git", "clone", clone_url, repo_dir], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository {repo_name}: {e}")
            return False
    else:
        print(f"Repository {repo_name} already exists at {repo_dir}.")

    venv_env = setup_repo_env(repo_dir)

    if not venv_env:
        return False
    print("Running test suite with pytest...")
    start_time = time.time()
    try:
        subprocess.run(
            ["./venv/bin/pytest"], cwd=repo_dir, check=True, env=venv_env, timeout=40
        )
    except subprocess.TimeoutExpired:
        print(f"Test suite timed out for {repo_name}.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Tests failed in repository {repo_name}: {e}")
        return False

    duration = time.time() - start_time
    if duration > MAX_TEST_DURATION:
        print(f"Test suite ran too slowly ({duration:.2f} seconds) for {repo_name}.")
        return False
    else:
        print(
            f"Test suite ran for {duration:.2f} seconds. Repository {repo_name} is accepted."
        )
        return True


def process_repo(repo):
    """
    Wraps clone_and_setup_repo so that we can return both the repository's full name
    and the result (True/False) for use in the callback.
    """
    result = clone_and_setup_repo(repo)
    base_dir = os.path.expanduser(repos_dir)
    repo_dir = os.path.join(base_dir, repo["name"])
    if not result:
        print("Delete and cleanup", cleanup_repo_dir(repo_dir))
    return (repo["full_name"], result)


def update_result(future):
    """
    Callback function that is called when a repository processing task completes.
    It updates the global saved_repo_count and writes out results to disk.
    """
    global saved_repo_count
    try:
        full_name, result = future.result()
        if result:
            with saved_repo_count_lock:
                saved_repo_count += 1
            with open("saved_repos.txt", "a") as f:
                f.write(f"{full_name}\n")
        else:
            failed_cache.add(full_name)
            save_failed_cache(failed_cache)
    except Exception as e:
        print(f"Exception processing repo: {e}")


def main():
    global saved_repo_count
    load_failed_cache()

    # Read list of already saved repositories.
    if os.path.exists("saved_repos.txt"):
        with open("saved_repos.txt", "r") as f:
            existing_list = set(line.strip() for line in f.readlines())
    else:
        existing_list = set()

    query = "language:Python stars:<2100"
    search_url = "https://api.github.com/search/repositories"
    per_page = 30
    page = 1

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Continue fetching pages and submitting tasks until we've reached our target.
        while saved_repo_count < NUM_REPOS_TO_SCRAPE:
            params = {"q": query, "per_page": per_page, "page": page}
            print(f"Fetching page {page}...")
            response = requests.get(search_url, params=params, headers=HEADERS)
            if response.status_code != 200:
                print(
                    "Error fetching repositories:", response.status_code, response.text
                )
                break

            results = response.json()
            repositories = results.get("items", [])
            if not repositories:
                print("No more repositories found.")
                break

            for repo in repositories:
                full_name = repo["full_name"]
                if full_name in failed_cache or full_name in existing_list:
                    print(f"Skipping {full_name} (cached).")
                    continue

                owner = repo["owner"]["login"]
                repo_name = repo["name"]
                stars = repo["stargazers_count"]
                print(f"\nChecking {owner}/{repo_name} ({stars} stars)...")

                print("  -> Repository meets criteria!")
                print(f"     URL: {repo['html_url']}")

                # Submit processing task and attach callback.
                future = executor.submit(process_repo, repo)
                future.add_done_callback(update_result)
                futures.append(future)

                # Check if we've reached our target.
                with saved_repo_count_lock:
                    if saved_repo_count >= NUM_REPOS_TO_SCRAPE:
                        print(f"\nReached target of {saved_repo_count} repositories.")
                        break

            if saved_repo_count >= NUM_REPOS_TO_SCRAPE:
                break

            page += 1
            time.sleep(1)  # Pause briefly between page fetches.

        # Wait for all submitted tasks to finish.
        concurrent.futures.wait(futures)

    print(f"\nScraped {saved_repo_count} repositories. Exiting.")


if __name__ == "__main__":
    global repos_dir
    repos_dir = sys.argv[1]
    main()
