import shutil
import signal
import subprocess
import time
import warnings
from urllib.parse import urljoin, urlparse

import pytest
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

MKDOCS_PORT = 8000
MKDOCS_URL = f"http://127.0.0.1:{MKDOCS_PORT}"
MKDOCS_HOME = "PytorchImagePipeline"


@pytest.fixture(scope="module", autouse=True)
def start_mkdocs_server():
    """
    Start the MkDocs server as a subprocess and ensure it's running before proceeding.
    """
    uv_full_path = shutil.which("uv")

    process = subprocess.Popen(  # noqa: S603
        [uv_full_path, "run", "mkdocs", "serve", "--dev-addr", f"127.0.0.1:{MKDOCS_PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)  # Give the server time to start

    # Verify that the server is running
    try:
        response = requests.get(MKDOCS_URL, timeout=10)
        response.raise_for_status()
    except Exception as e:
        kill_mkdocs_server(process)
        pytest.fail(f"MkDocs server failed to start: {e}")

    yield

    # Teardown: stop the server
    kill_mkdocs_server(process)


def kill_mkdocs_server(process):
    try:
        # Send CTRL+c to kill the child process
        process.send_signal(signal.SIGINT)
    except subprocess.TimeoutExpired:
        pytest.fail("Timeout occured")
    process.terminate()
    process.wait()


def get_navigation_links():
    """
    Fetch all navigation links from the MkDocs homepage.
    """
    response = requests.get(MKDOCS_URL, timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")
    nav = soup.find("nav", class_="md-nav md-nav--primary")
    ul = nav.find("ul")
    nav_links = extract_primenav_hrefs(ul)
    return nav_links


def extract_primenav_hrefs(ul):
    hrefs = []
    for li in ul.find_all("li", recursive=False):
        if "md-nav__item--nested" in li.get("class", []):
            # If the <li> is nested, find the <ul> inside and extract hrefs
            nested_ul = li.find("ul")
            if nested_ul:
                hrefs.extend(extract_primenav_hrefs(nested_ul))
        else:
            # If the <li> is not nested, extract the href from the <a> tag
            a_tag = li.find("a", href=True)
            if a_tag:
                hrefs.append(a_tag["href"].replace(".", "/"))
    return hrefs


def get_page_links(page_url):
    """
    Fetch all links (internal and external) from a given page.
    """
    if page_url != "/":
        page_url = "/" + page_url
    response = requests.get(urljoin(MKDOCS_URL, MKDOCS_HOME + page_url), timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")
    page_content = soup.find("article", class_="md-content__inner md-typeset")

    links = []
    for a in page_content.find_all("a", href=True):
        link = urljoin(page_url, a["href"])
        links.append(link)

    return links


def is_internal_link(link):
    """
    Determine if a link is internal (relative to the MkDocs site).
    """
    parsed = urlparse(link)
    return parsed.netloc == "" or parsed.netloc.startswith("127.0.0.1")


def get_page_results():
    nav_links = get_navigation_links()

    broken_internal_links = []
    broken_external_links = []

    for page_url in nav_links:
        links = get_page_links(page_url)

        for link in links:
            if link.startswith("/"):
                link = urljoin(MKDOCS_URL, MKDOCS_HOME + link, True)
            try:
                response = requests.get(link, timeout=5)
                if is_internal_link(link):
                    if response.status_code != 200:
                        broken_internal_links.append({
                            "Page": page_url,
                            "Link": link,
                            "Code": response.status_code,
                        })
                else:
                    if response.status_code != 200:
                        broken_external_links.append({
                            "Page": page_url,
                            "Link": link,
                            "Code": response.status_code,
                        })
            except requests.RequestException:
                if is_internal_link(link):
                    broken_internal_links.append({"Page": page_url, "Link": link, "Code": 418})
                else:
                    broken_external_links.append({"Page": page_url, "Link": link, "Code": 418})
    return broken_internal_links, broken_external_links


def test_mkdocs_links():
    """
    Test all pages in navigation for valid internal and external links.
    """
    broken_internal_links, broken_external_links = get_page_results()

    # Log warnings for broken external links
    if broken_external_links:
        print("\nWarnings for broken external links:")
        print(tabulate(broken_external_links, headers="keys", tablefmt="grid"))
        warnings.warn(UserWarning("Warnings for broken external links. See table below."), stacklevel=2)

    # Fail the test if there are broken internal links
    if broken_internal_links:
        print("\nFailed internal links:")
        print(tabulate(broken_internal_links, headers="keys", tablefmt="grid"))
        assert not broken_internal_links, "Broken internal links found. See table above."
