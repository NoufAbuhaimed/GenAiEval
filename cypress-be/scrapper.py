import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
import os

# Function to fetch the webpage content with retries
def fetch_webpage(url, retries=3, timeout=5):
    for i in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
            return response.content
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"Attempt {i + 1} failed with error: {e}")
            time.sleep(2 ** i)  # Exponential backoff
    raise Exception(f"Failed to retrieve the webpage after {retries} attempts")

# Function to get all links from a webpage
def get_all_links(url, visited):
    links = []
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")
        return links

    soup = BeautifulSoup(response.content, 'html.parser')
    base_url = "{0.scheme}://{0.netloc}".format(urllib.parse.urlsplit(url))

    for link in soup.find_all('a', class_='no-underline'):
        href = link.get('href')
        if href and not href.startswith('#'):
            full_url = urllib.parse.urljoin(base_url, href)
            if full_url not in visited:
                visited.add(full_url)
                links.append(full_url)
                links += get_all_links(full_url, visited)  # Recursively fetch links from the current page

    return links

# Main function to scrape and save data
def scrape_and_save(url, file_path):
    try:
        # Fetch the content of the webpage
        webpage_content = fetch_webpage(url)

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(webpage_content, 'html.parser')

        # Extract data: for example, extract all paragraph texts
        paragraphs = soup.find_all('p')
        data = [p.get_text() for p in paragraphs]

        # Save data to the specified file
        with open(file_path, 'a', encoding='utf-8') as f:  # Use 'a' for append mode
            f.write(f"URL: {url}\n")
            f.write("Paragraphs:\n")
            for paragraph in data:
                f.write(paragraph + '\n')
            f.write('\n')

        print(f"Data scraped and saved successfully for {url}.")
    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")

start_url = "https://u.ae/en/information-and-services#/"
visited_links = set()
output_file = r"C:\Users\User\Downloads\cypress\cypress-be\scraped_data.txt"

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Get all links starting from the start_url
all_links = get_all_links(start_url, visited_links)

if all_links:  # Ensure there are links in the list before deleting
    del all_links[-1]  # Delete the last element

# Scrape and save data for each URL
for url in all_links:
    scrape_and_save(url, output_file)
